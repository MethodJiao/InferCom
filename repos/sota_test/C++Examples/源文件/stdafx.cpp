// stdafx.cpp : 只包括标准包含文件的源文件
// Examples.pch 将作为预编译头
// stdafx.obj 将包含预编译类型信息

#include "stdafx.h"


#pragma comment (lib, "P3DDC.lib")
#pragma comment (lib, "P3DGeomBase.lib")
#pragma comment (lib, "P3DGeomObject.lib")
#pragma comment (lib, "BPDataCore.lib")
#pragma comment (lib, "TgGe.lib")
#pragma comment (lib, "BPPlatform.lib")
#pragma comment (lib, "BPSolidCore.lib")
#pragma comment (lib, "BPDataExt.lib")
#pragma comment (lib, "JsonCpp.lib")
#pragma comment (lib, "BPDrawingCommon.lib")
#pragma comment (lib, "BPUIFrameWork.lib")
#pragma comment (lib, "BPApp.lib")
#pragma comment (lib, "BPApp.lib")
#pragma comment (lib, "BPMaterialConfig.lib")
#pragma comment (lib, "BPDataCore.lib")
#pragma comment (lib, "BPParametricBase.lib")
#pragma comment (lib, "BPPrimaryElement.lib")
#pragma comment (lib, "BPPluginManager.lib")
#pragma comment (lib, "BPCoordInput.lib")
#pragma comment (lib, "BPProjectTree.lib")


#ifdef DEBUG
#pragma comment (lib, "WDUiD.lib")
#else
#pragma comment (lib, "WDUi.lib")
#endif