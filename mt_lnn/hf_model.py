"""
mt_lnn/hf_model.py — HuggingFace-native packaging for the O-series.

WHY THIS FILE EXISTS
--------------------
serve/ 是自建服务栈（server.py 走 MTLNNModel + torch.load，server_hf.py 走
"冻结 HF 基座 + adapter"）。两条线都没有 config.json、都没有
from_pretrained，因此进不了 vLLM / SGLang 的 Auto 类分发，也进不了 Hub 上
"pip 装完三行代码"的使用习惯。本文件把 ``mt_lnn.model.MTLNNModel`` 包成
transformers 风格的 ``PreTrainedModel``，让 O-series 可以：:

    from mt_lnn.hf_model import MTLNNForCausalLM
    model = MTLNNForCausalLM.from_pretrained("org/mtlnn-o1")
    ids = model.generate(ids, max_new_tokens=32, do_sample=False)

DESIGN RULES
------------
1. **数学零改动**。forward / generate 全部委托给 ``MTLNNModel``，本文件只做
   两件事：把 ``dict`` 输出包成 ``CausalLMOutputWithPast``、把
   ``MTLNNConfig`` 与 ``PretrainedConfig`` 互相翻译。没有任何新的数值路径，
   因此"加载路径输出 == 直接构造输出"是构造上成立的，而不是靠测试兜住。
2. **config.json 必须能独立重建计算图**。它携带
   ``product_line``（"O"=原生 MTLNNModel / "M"=基座+adapter）+ 原生
   ``MTLNNConfig`` 的全部 init 字段 + ``graph``（重建所需的图开关）。这与
   serve/server_hf.py "从 checkpoint 的 args 重建出与训练时一致的图"是同一
   套语义，只是宿主从 torch.load 换成 transformers 的序列化通道。
3. **不新增第三方依赖**。transformers 已在 requirements.txt。另外这里**刻意不
   导出到 ``mt_lnn/__init__.py``**：那样会让 ``import mt_lnn`` 无条件拉起
   transformers，把 ONNX / 自建 serve 这些不需要它的路径也拖下水。用户显式
   ``from mt_lnn.hf_model import MTLNNForCausalLM`` 即可。

KNOWN BOUNDARIES（诚实标注，不是 TODO）
---------------------------------------
* **不走 HF 的 GenerationMixin.generate**。MTLNNModel 的缓存是
  ``ModelCacheStruct``（循环状态 h_prev + KV + coherence KV），HF 的
  ``DynamicCache`` 表达不了它；硬接 prepare_inputs_for_generation 只会改数
  学。这里覆写 ``generate`` 委托原生实现，代价是 GenerationConfig 的 beam
  search / constrained decoding 等策略不可用（原生 generate 支持
  greedy / temperature / top_k / top_p / EOS / 回调）。
* **不在 meta device 上建图**。transformers **5.x 的 from_pretrained 一律用
  ``torch.device("meta")`` 构造模型**（4.x 的 ``low_cpu_mem_usage`` 开关已
  被移除并且不再生效），加载完再用 ``torch.empty_like`` 把 non-persistent
  buffer 搬回 CPU —— **值是未初始化的垃圾**。库自带的重算只认 class 名含
  "RotaryEmbedding" 且带 ``original_inv_freq`` 的模块，MTLNNModel 的 RoPE
  表、注意力距离掩码、GWTB causal mask 都不在其列。这里用两道措施兜住：
  ``get_init_context`` 摘掉 meta 上下文（等价 4.x 的 eager 建图），
  ``_init_weights`` 再按 ``reset_non_persistent_buffers()`` 协议重算一遍。
  代价是加载峰值内存多一份权重；换来的是"加载出来的模型一定等于构造出来的"。
* **权重绑定**。``MTLNNConfig.tie_embeddings=True`` 时 lm_head.weight 与
  embedding.token_embed.weight 是同一个 Parameter，safetensors 不允许别名
  张量，故声明 ``_tied_weights_keys``：保存时只留一份，加载时由
  ``tie_weights()`` 重新绑定。untied 时不存在别名，且 5.x 的
  ``get_expanded_tied_weights_keys`` 会被 ``config.tie_word_embeddings``
  挡掉，所以同一份声明对两种配置都安全。
* **不要调用 ``post_init()``**。5.x 的 ``post_init()`` 末尾会跑
  ``init_weights()``，那会把 ``from_mtlnn_model`` 包进来的训练好的权重重新
  初始化掉。
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .config import MTLNNConfig
from .model import MTLNNModel

try:
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError as exc:  # pragma: no cover - 依赖缺失属于环境问题
    raise ImportError(
        "mt_lnn.hf_model 需要 transformers>=4.40（requirements.txt 已声明）。"
        "O-series 的其余部分不依赖 transformers。"
    ) from exc


__all__ = [
    "MODEL_TYPE",
    "PRODUCT_LINE_O",
    "PRODUCT_LINE_M",
    "MTLNNConfigHF",
    "MTLNNForCausalLM",
    "tiny_o_config",
    "tiny_o_model",
    "register_for_auto_classes",
]

MODEL_TYPE = "mtlnn"
PRODUCT_LINE_O = "O"          # 原生 MTLNNModel（端侧 / attention-free 线）
PRODUCT_LINE_M = "M"          # 冻结 HF 基座 + MT adapter（PEFT 式分发线）
# O 系列的"图重建"就是把 mtlnn_config 原样喂给这个类 —— 写进 config.json，
# 让拿到目录的人不必猜。非空也保证它不会被 to_diff_dict() 当成默认值丢掉。
O_GRAPH_BUILDER = "mt_lnn.model:MTLNNModel"

# trust_remote_code 的入口声明：写进 config.json 后，Hub 侧无需 pip 安装本仓
# 也能解析（前提是 mt_lnn 已在环境中）。主路径仍然是显式 import。
DEFAULT_AUTO_MAP = {
    "AutoConfig": "mt_lnn.hf_model.MTLNNConfigHF",
    "AutoModelForCausalLM": "mt_lnn.hf_model.MTLNNForCausalLM",
}

# 只在 __init__ 里可传的原生字段：d_proto / d_proto_total 由 __post_init__
# 推导（field(init=False)），不能回灌，否则 MTLNNConfig 会 TypeError。
_NATIVE_INIT_FIELDS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(MTLNNConfig) if f.init
)


class MTLNNConfigHF(PretrainedConfig):
    """``MTLNNConfig`` 的 transformers 版本。

    序列化布局::

        {
          "model_type": "mtlnn",
          "product_line": "O",
          "mtlnn_config": { ...原生 MTLNNConfig 的 init 字段... },
          "graph":        { ...图重建开关（O 系列留空 / M 系列见 adapter_export）... },
          "auto_map":     { ... }
        }

    之所以把原生字段整体嵌一层而不是摊平到顶层：MTLNNConfig 有 60+ 个字段且
    随架构分支持续增删，摊平意味着每加一个字段就要同步改两处。嵌一层后，
    ``to_mtlnn_config()`` 用 dataclass 反射过滤，字段增删零维护。
    """

    model_type = MODEL_TYPE

    def __init__(
        self,
        product_line: str = PRODUCT_LINE_O,
        mtlnn_config: Optional[Dict[str, Any]] = None,
        graph: Optional[Dict[str, Any]] = None,
        auto_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.product_line = str(product_line).upper()
        self.mtlnn_config = dict(mtlnn_config or {})
        self.graph = dict(graph or {})
        self.auto_map = dict(auto_map or DEFAULT_AUTO_MAP)

    # -- 与原生 config 互转 -------------------------------------------------
    def to_mtlnn_config(self) -> MTLNNConfig:
        """序列化字典 -> ``MTLNNConfig``（未知键静默丢弃，保证前向兼容）。"""
        known = {k: v for k, v in self.mtlnn_config.items() if k in _NATIVE_INIT_FIELDS}
        return MTLNNConfig(**known)

    @classmethod
    def from_mtlnn_config(
        cls,
        native: MTLNNConfig,
        product_line: str = PRODUCT_LINE_O,
        graph: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "MTLNNConfigHF":
        payload = {k: getattr(native, k) for k in _NATIVE_INIT_FIELDS}
        return cls(product_line=product_line, mtlnn_config=payload,
                   graph=graph if graph is not None else {"builder": O_GRAPH_BUILDER},
                   **kwargs)

    @property
    def is_m_series(self) -> bool:
        return self.product_line == PRODUCT_LINE_M


class MTLNNForCausalLM(PreTrainedModel):
    """O-series 的 causal LM：``MTLNNModel`` + transformers 序列化外壳。

    权重全部住在 ``self.model``（``base_model_prefix``）里，文件布局与标准
    HF 仓库一致（config.json + model.safetensors）。
    """

    config_class = MTLNNConfigHF
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    main_input_name = "input_ids"
    # 5.x 的 _tied_weights_keys 是 {目标: 来源} 字典（4.x 是列表），且只在
    # config.tie_word_embeddings 为真时生效 —— 见文件头 KNOWN BOUNDARIES 第 3 条。
    _tied_weights_keys = {
        "model.lm_head.weight": "model.embedding.token_embed.weight",
    }

    def __init__(self, config: MTLNNConfigHF, model: Optional[MTLNNModel] = None):
        """``model`` 给了就直接包裹（零拷贝、不重建），否则按 config 新建。"""
        super().__init__(config)
        self.model = model if model is not None else MTLNNModel(config.to_mtlnn_config())
        # tie_weights() 与保存时的别名剔除都看这个开关；必须反映原生配置。
        self.config.tie_word_embeddings = bool(self.model.config.tie_embeddings)
        # 基类 __init__ 已经在设置开关之前算过一次，这里必须按最终值重算。
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)

    # -- 嵌入层访问器（tie_weights / resize_token_embeddings 依赖） ---------
    def get_input_embeddings(self):
        return self.model.embedding.token_embed

    def set_input_embeddings(self, value):
        self.model.embedding.token_embed = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    # -- 前向：dict -> ModelOutput，一行数值代码都没有 ----------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        """委托 ``MTLNNModel.forward``。

        注意两点映射：HF 的 ``attention_mask`` 是 0/1 整型，这里转成 bool
        ``pad_mask``；``past_key_values`` 收/发的是 **单元素 tuple**，
        ``out.past_key_values[0]`` 才是 ``ModelCacheStruct``（原因见
        ``_pack_cache``）。
        """
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            cache=_unpack_cache(past_key_values),
            pad_mask=_as_bool_pad_mask(attention_mask),
            use_cache=bool(use_cache) if use_cache is not None else False,
        )
        loss = out.get("loss")
        if return_dict is False:
            return (loss, out["logits"])
        return CausalLMOutputWithPast(
            loss=loss,
            logits=out["logits"],
            past_key_values=_pack_cache(out.get("cache")),
        )

    # -- 生成：委托 MTLNNModel.generate（见文件头 KNOWN BOUNDARIES） --------
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        cache: Optional[Any] = None,
        return_cache: bool = False,
        step_callback: Optional[Callable[[Any, int], None]] = None,
        **hf_generation_kwargs,
    ):
        """与 ``MTLNNModel.generate`` 同签名；额外接受 HF 的 ``max_length``。

        HF 侧其余 Generate 参数（num_beams、forced_decoder_ids …）被忽略：
        原生解码器不实现它们，静默接受比给出错误结果更危险，故在
        docs/SERVING_ROADMAP.md 中显式列出这一缺口。
        """
        if "max_length" in hf_generation_kwargs and input_ids is not None:
            max_new_tokens = int(hf_generation_kwargs["max_length"]) - input_ids.shape[-1]
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            cache=cache,
            return_cache=return_cache,
            step_callback=step_callback,
        )

    # -- 权重初始化：5.x 会在加载后回调这里，我们用它重算 non-persistent buffer
    # -------------------------------------------------------------------------
    @classmethod
    def get_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels):
        """摘掉 meta device 上下文（等价 4.x 的 ``low_cpu_mem_usage=False``）。

        见文件头 KNOWN BOUNDARIES 第 2 条。摘掉之后模型在真实设备上构造，
        __init__ 里算出来的 buffer 都是有效值；加载链路仍然会把它们用
        ``torch.empty_like`` 覆写成垃圾，所以这一步必须和下面的
        ``_init_weights`` 配套使用，单靠其中一个都不够。
        """
        contexts = super().get_init_context(
            dtype, is_quantized, _is_ds_init_called, allow_all_kernels
        )
        return [c for c in contexts
                if not (isinstance(c, torch.device) and c.type == "meta")]

    def _init_weights(self, module) -> None:
        """重算 non-persistent buffer；其余交给库（已加载的参数会被 init 守卫跳过）。

        transformers 5 的 ``initialize_weights()`` 会对每个模块回调这里。模块
        只要实现了 ``reset_non_persistent_buffers()``（值 == __init__ 构造出来
        的值）就会被重算；没实现的模块（纯参数 / 纯诊断）不受影响。
        """
        super()._init_weights(module)
        reset = getattr(module, "reset_non_persistent_buffers", None)
        if callable(reset):
            reset()

    # -- 构造 / 加载 --------------------------------------------------------
    @classmethod
    def from_mtlnn_model(
        cls,
        model: MTLNNModel,
        product_line: str = PRODUCT_LINE_O,
        graph: Optional[Dict[str, Any]] = None,
    ) -> "MTLNNForCausalLM":
        """把既有的 ``MTLNNModel``（训练产物 / serve 载入的 checkpoint）包起来。

        **零拷贝**：``self.model`` 直接指向传入对象，不重建、不复制权重，所以
        包装后的前向输出与包装前逐位相同。
        """
        cfg = MTLNNConfigHF.from_mtlnn_config(model.config, product_line, graph)
        return cls(cfg, model=model)



def register_for_auto_classes() -> None:
    """把本模块的类注册进 transformers 的 AUTO 映射（幂等）。

    注册后 ``AutoModelForCausalLM.from_pretrained(path)`` 无需 trust_remote_code
    即可工作——前提是调用方先 import 过本模块（这是 OOT 架构的标准代价）。
    """
    try:
        AutoConfig.register(MODEL_TYPE, MTLNNConfigHF, exist_ok=True)
        AutoModelForCausalLM.register(MTLNNConfigHF, MTLNNForCausalLM, exist_ok=True)
    except TypeError:  # transformers<4.33 的 register 没有 exist_ok
        if MODEL_TYPE not in AutoConfig._model_mapping:
            AutoConfig.register(MODEL_TYPE, MTLNNConfigHF)


def tiny_o_config(**overrides) -> MTLNNConfig:
    """随机小权重口径的 O 系列配置（测试与基准脚本共用的关键设计件）。

    形状取自 tests/test_generate.py 的 proven-tiny 口径
    （d_model=104 = 13 protofilaments × d_proto 8，d_proto 是 8 的倍数所以
    不会触发 Tensor-Core 对齐告警），并按文档里 O 系列的语义关掉 GWTB
    （它预分配 O(T²) causal mask，是 O 系列唯一残留的二次项）。

    规模：~2 层、vocab 64，构造 + 前向在 CPU 上是毫秒级，整套测试预算 <2min。
    """
    base: Dict[str, Any] = dict(
        vocab_size=64,
        max_seq_len=32,
        d_model=104,
        n_layers=2,
        n_heads=13,
        n_kv_heads=1,
        d_head=8,
        n_protofilaments=13,
        map_hidden_dim=8,
        n_time_scales=2,
        gwtb_n_heads=1,
        dropout=0.0,
        attention_dropout=0.0,
        use_gwtb=False,          # O 系列：无 O(T²) workspace mask
    )
    base.update(overrides)
    return MTLNNConfig(**base)


def tiny_o_model(seed: int = 0, **overrides) -> MTLNNForCausalLM:
    """``tiny_o_config`` + 固定种子的随机小权重 O 系列模型（eval 模式）。"""
    torch.manual_seed(seed)
    wrapper = MTLNNForCausalLM.from_mtlnn_model(
        MTLNNModel(tiny_o_config(**overrides)),
        product_line=PRODUCT_LINE_O,
    )
    return wrapper.eval()


def _as_bool_pad_mask(attention_mask: Optional[torch.Tensor]):
    """HF 的 attention_mask 是 (B, T) 的 0/1 int，MTLNNModel 要 bool 掩码。"""
    if attention_mask is None:
        return None
    return attention_mask if attention_mask.dtype == torch.bool else attention_mask.bool()


def _pack_cache(cache: Optional[Any]):
    """把 ``ModelCacheStruct`` 装进单元素 tuple 再交给 ModelOutput。

    ``ModelOutput.__post_init__`` 会对 ``past_key_values`` 做 ``tuple(v)``；
    ``ModelCacheStruct`` 是普通类（不可迭代），直接传会在部分 transformers
    版本上抛 TypeError。单元素 tuple 让这个转换变成恒等操作，代价是取值要
    多一层 ``[0]``。
    """
    return None if cache is None else (cache,)


def _unpack_cache(past_key_values: Optional[Any]):
    """``_pack_cache`` 的逆操作；同时容忍调用方直接传裸 ``ModelCacheStruct``。"""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 1:
        return past_key_values[0]
    return past_key_values


register_for_auto_classes()
