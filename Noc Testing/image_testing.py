import glob
import os


rootDir = '.'

for dirName, subDirlist, fileList in os.walk(rootDir):
    print(dirName)
    print(subDirlist)
    print(fileList)
