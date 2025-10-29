// stdafx.h : 标准系统包含文件的包含文件，

#pragma once

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN            // 从 Windows 头中排除极少使用的资料
#endif

#include "targetver.h"

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // 某些 CString 构造函数将是显式的

#include <afxwin.h>         // MFC 核心组件和标准组件
#include <afxext.h>         // MFC 扩展

#ifndef _AFX_NO_OLE_SUPPORT
#include <afxole.h>         // MFC OLE 类
#include <afxodlgs.h>       // MFC OLE 对话框类
#include <afxdisp.h>        // MFC 自动化类
#endif // _AFX_NO_OLE_SUPPORT

#ifndef _AFX_NO_DB_SUPPORT
#include <afxdb.h>                      // MFC ODBC 数据库类
#endif // _AFX_NO_DB_SUPPORT

#ifndef _AFX_NO_DAO_SUPPORT
#include <afxdao.h>                     // MFC DAO 数据库类
#endif // _AFX_NO_DAO_SUPPORT

#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>           // MFC 对 Internet Explorer 4 公共控件的支持
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>                     // MFC 对 Windows 公共控件的支持
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <afxcontrolbars.h>     // 功能区和控件条的 MFC 支持

#include "BcgFrame/BCGCBProInc.h"
#include <boost/integer.hpp>
#include <boost/integer_fwd.hpp>
#include <boost/bimap.hpp>

//////////////////////////////////////////////////////////////////////////
#include "resource.h"
#include <vector>
#include <map>
#include <algorithm>
#ifndef PI
#define PI ((double)3.1415926535897932)
#endif
using namespace std;

// 图形平台基础数据头文件
#include "BPExportMacros.h"
#include "PKPMExportMacros.h"
#include "P3DDC/P3DDCAPI.h"
#include "P3DGeomObject/P3DGeomObjectPublicAPI.h"

// BIMBase Publish API头文件
#include "JsonCpp/JsonCppAPI.h"
#include "BPBase/BPBaseAPI.h"
#include "BPData/BPDataAPI.h"
#include "BPDataCore/BPDataCoreDef.h"
#include "BPDataCore/BPDataCoreAPI.h"
#include "BPDataCore/BPPlacement.h"
#include "BPDataCore/BPDomainDataInitManager.h"
#include "BPDataCore/BPEntitySymbologyEvent.h"
#include "BPDisplay/BPDisplayAPI.h"
#include "BPInteraction/BPInteractionAPI.h"
#include "BPFrame/BPFrameAPI.h"
#include "BPSolidCore/BPSolidCoreAPI.h"
#include "BPBase/BPPlatformAPI.h"

// 专业平台头文件
#include "TgGe/TgGe.h"
#include "PBClashDetection/PBClashDetectionAPI.h"
#include "PBClashDetection/ClashDetection.h"
#include "PBBimCore/PBBimCoreAPI.h"
#include "PBBimCore/PBAxisElement.h"
#include "BPApp/PBBimAppAPI.h"
#include "PBBimCore/PBBimSchemaManager.h"
#include "PBPlatform/PBPlatformAPI.h"
#include "PBBimCore/PBClippingEvent.h"
#include "PBBimUtilities/rapidjson/document.h"
#include "PBBimUtilities/rapidjson/stringbuffer.h"
#include "PBBimUtilities/rapidjson/Writer.h" 
#include "BPUIFrameWork/BPUIFrameWorkAPI.h"
#include "BPUIFrameWork/BPUIFrameWorkUtil.h"
#include "BPUIFrameWork/BPNewDockContainer.h"
#include "BPUIFrameWork/BPNewDockManager.h"
#include "BPUIFrameWork/RButtonClickItem.h"
#include "BPUIFrameWork/BPViewRButtonClickListener.h"
#include "BPUIFrameWork/BPViewRButtonClickListenerCenter.h"
#include "BPApp/PBToolSettingManager.h"
#include "PBBimUtilities/PBimsErrExceptionUtil.h"
#include "PBBimTools/PBBimToolsAPI.h"
#include "BPPrimaryElement/BPHatchVardef.h"
#include "BPPrimaryElement/BPPrimaryGeomAPI.h"
#include "BPPrimaryElement/BPHatchDefine.h"
#include "BPPrimaryElement/BPHatch.h"
#include "BPPrimaryElement/BPHatchPattern.h"
#include "BPPrimaryElement/BPHatchPatternManager.h"

//资源切换头文件
#include "PBBimCore/PBBimExtensionModule.h"
#include "PBBimCore/PBBimCoreAPI.h"
#include "BPUIFrameWork/BPNewDockManager.h"
#include "BPUIFrameWork/BPNewDockContainer.h"
#include "cereal/archives/binary.hpp"
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#define DefineSuper(B)  public: typedef B T_Super; 
using namespace PBBim;
using namespace PBBim::PBBimCore;
using namespace BIMBase;
using namespace P3DApi::P3DApplication;
using namespace BIMBase::Data;
using namespace BIMBase::Core;
using namespace BIMBase::SolidCore;

//材质
#include"MaterialConfig/BPMaterialEx.h"

//数据交换
#include "DataExchange/DataExchangeAPI.h"

//属性面板相关
#include "Share/Common/WDPublicDef.h"
#include"Share/Common/ExportMacro.h"
#include "WDUi/WDToolProperty.h"

//追踪器相关
#include "Common/SDDef.h"
#include "BPCoordInput/BPCoordInputAPI.h"
#include "TracerCusFunDemo.h"

//施工图相关
#include "BPPrimaryElement/BPTextEntity.h"
#pragma comment(lib,"BPSerialCommonLib.lib")

//组件
#pragma comment(lib,"BPParametricData.lib")
#pragma comment(lib,"BPParametricCore.lib")
#pragma comment(lib,"BPParametricBase.lib")
#pragma comment(lib,"BPParaComponent.lib")

#include "BPParametricData/BPParametricDataAPI.h"
#include "BPParametricCore/BPParametricCoreAPI.h"
#include "BPParametricBase/BPParametricBaseAPI.h"
//#include "BPParametricComponent/BPParametricComponentAPI.h"
#include "BPParaComponent/BPParametricComponentAPI.h"



//项目树
#include "BPProjectTree/BPProjectTreeDef.h"
#include "BPProjectTree/BPProjectTree.h"
#include "BPProjectTree/BPProjectTreeManager.h"

//线程
#include "BPThreads/BPThreads.h"
//创建工作线程
#include "BPUIFrameWork/BPAsynchronousUtil.h"
//范例相关
#include "resource.h"
#include "ExampleDef.h"
#include "BPDrawingParasManagerDemo.h"
#include "IBPDrawingCenterDemo.h"
#include "BPDemoGraphicElementDemo.h"
#include "CubeDemo.h"
#include "DrainDemo.h"
#include "EmbankmentDemo.h"
#include "SolidDemo.h"
#include "OpenningDemo.h"
#include "ToolLayoutCubeDemo.h"
#include "DlgLayoutLineDemo.h"
#include "BaseDataDemo.h"
#include "LineDemo.h"
#include "CrossArmDemo.h"
#include "TowerDemo.h"
#include "UniversalBeamDemo.h"
#include "DataManagerDemo.h"
#include "TubeDemo.h"
#include "IndependentBridgeDemo.h"
#include "ToolLayoutIndependentBridgeDemo.h"
#include "ExampleTubeDemo.h"
#include "ToolLayoutExampleTubeDemo.h"
#include "ToolExampleTubeCopyDemo.h"
#include "ToolExampleTubeMoveDemo.h"
#include "ToolExampleTubeDeleteDemo.h"
#include "ToolExampleTubeRotateDemo.h"
#include "RButtonClickEventDemo.h"
#include "IndependentBridgePropertyDemo.h"
#include "TransactionDemo.h"


static void serializePoint3DVec(cereal::BinaryOutputArchive& ar, const std::vector<GePoint3d>& pt)
{
	ar(pt.size());
	for (GePoint3d p : pt)
	{
		ar(p.x, p.y, p.z);
	}

}
static void deSerializePoint3DVec(cereal::BinaryInputArchive& ar, std::vector<GePoint3d>& vec)
{

	vec.clear();
	size_t size = 0;
	ar(size);
	for (int i = 0; i < size; i++)
	{
		double x, y, z;
		ar(x, y, z);
		GePoint3d pt = GePoint3d::create(x, y, z);
		vec.push_back(pt);
	}

}