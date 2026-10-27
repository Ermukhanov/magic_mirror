Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Acer Nitro\Downloads\magic_mirror"
WshShell.Run "pythonw magic_mirror_v2.py", 0, False
