import laspy

path = r"D:\Raphael\Essais\1_TUILES_DE_REFERENCE\2563000_1208500.copc.laz"

pc = laspy.read(path)

if pc.vlrs is not None:
    pc.header.vlrs.extract("CopcInfoVlr")

if pc.evlrs is not None:
    pc.evlrs.extract("CopcHierarchyVlr")

pc.write(r"C:\Users\wurglerr\Downloads\2563000_1208500.laz")