import sys
sys.path.insert(0, ".")
import paramiko

HOST = "65.49.232.182"; PORT = 10012
USER = "user"; PASSWORD = "DD6ydue8leCNT/nA"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASSWORD, timeout=30)
sftp = c.open_sftp()
sftp.get("/home/user/M1/benchmarks/efficiency_curve_full.json",
         "benchmarks/efficiency_curve_full.json")
sftp.get("/home/user/M1/benchmarks/efficiency_curve_128k.json",
         "benchmarks/efficiency_curve_128k.json")
sftp.close()
c.close()
print("downloaded")
