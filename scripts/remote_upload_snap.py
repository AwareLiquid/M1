import sys
sys.path.insert(0, ".")
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("65.49.232.182", 10012, "user", "DD6ydue8leCNT/nA", timeout=30)
sftp = c.open_sftp()
for local, remote in [
    ("mt_lnn/config.py", "/home/user/M1/mt_lnn/config.py"),
    ("mt_lnn/mt_lnn_layer.py", "/home/user/M1/mt_lnn/mt_lnn_layer.py"),
]:
    sftp.put(local, remote)
    print(f"uploaded {local} -> {remote}")
sftp.close()
c.close()
print("done")
