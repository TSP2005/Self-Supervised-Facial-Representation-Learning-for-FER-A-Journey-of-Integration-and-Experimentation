from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

file = drive.CreateFile({'title': './ckpts/fer.pth.tar'})  
file.SetContentFile('./ckpts/fer.pth.tar')  
file.Upload()
print("File uploaded successfully!")
