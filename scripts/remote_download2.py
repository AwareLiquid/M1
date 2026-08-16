import sys
sys.path.insert(0, ".")
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("65.49.232.182", 10012, "user", "DD6ydue8leCNT/nA", timeout=30)
sftp = c.open_sftp()
sftp.get("/home/user/eval_seeds.log", "logs/eval_seeds_cloud.log")
sftp.get("/home/user/eval_p4.log", "logs/eval_p4_cloud.log")
sftp.close()
c.close()
print("downloaded eval logs")
