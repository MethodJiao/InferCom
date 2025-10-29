#include "stdafx.h"
//#include "ExampleDef.h"
//#include "Examples.h"
////#include "PBBimapp\PBToolSettingManager.h"
////#include "DlgTestSelectTool.h"
////#include "DlgLayoutLineTest.h"
////#include "DlgLayoutCubeTest.h"
//#include "ProjectEventTest.h"
////#include "DlgDockTest.h"
////#include "DlgSolidType.h"
////#include "PBBimApp/PBToolSettingManager.h"
//#include "DomainChangeEvent.h"
//#include "EntitySymbologyEvent.h"
//#include "ToolClashDemo.h"
//#include "ObjectChangeEventTest.h"
//#include "adsarxfunc/ADSARXFUN.h"
//#include "MyCubeDragManipulatorTest.h"
//#include "CubeTest.h"
//#include "ElementChangeEventTest.h"
//#include "ObjectChangeEventTest.h"
//#include "BPUIFrameWork/BPToolBarUtil.h"

#pragma comment (lib, "adsarxfunc.lib")

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

static void PlaceBall()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;
	PModelId modelId = pModel->getModelId();
	if (!modelId.IsValid())
		return;
		
	GimGeBase::GimGeSphere commonModel;
	commonModel.setR(500);
	GePoint3d ptCenter=GePoint3d::create(1000, 0, 0);
	commonModel.setCenter(ptCenter);
	

	if (SUCCESS != commonModel.addToProject(*pProject, modelId))
		return;

	/*GeSphereInfo sphere(GePoint3d center, double radius);
	double radius = 500;
	GePoint3d center(double x = 1000, double y = 0, double z = 0);
	IGeSolidBasePtr createGeSphere(GeSphereInfo data);*/
	
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("placeBallTest", &PlaceBall);
AutoDoRegisterFunctionsEnd

#include "stdafx.h"

//测试命令：从指定路径导入bfa模型，然后对该类型对象设置一条类型属性和一条实例属性，并在（-1000,0,0）的位置创建一个实例对象布置下去
static void testImportComponent()
{
	std::wstring path = L"PLATFORM\\BPGroup\\ComponentLib\\变电\\电气\\变压器\\三绕组变压器\\110kV变压器模型15.bfa";
	std::wstring bfaPath = BIMBase::Core::BPApplication::getInstance().getAppPath().c_str() + path;
	std::string strBfaPath = wstring2string(bfaPath);
	std::vector<Data::BPObjectPtr> importObjs;
	//从指定路径导入bfa模型
	if (!GimGeBase::PEGlobalDLComponentPropManager::importBfa(strBfaPath, importObjs))
		return;
	Data::BPObjectPtr object = importObjs.at(0);
	if (!object.isValid())
		return;
	BPProjectP project = BPProject::getActiveProject();
	BIMBase::Core::BPModelBaseP model = project->getActiveModel();
	GimGeBase::PBDLComponentPtr dlComp = dynamic_cast<GimGeBase::PBDLComponentP>(object.get());
	if (!dlComp.isValid())
		return;
	//增加一个属性分组，如果为true，则这个属性组属于类型属性，false则为实例属性，增加属性分组或者增加属性条目需要对类型对象进行操作
	dlComp->addPropGroup(L"基本属性", true);
	std::wstring propName = L"名称";
	std::wstring propValue = L"110kv变压器模型";
	//属性条目设置:属性名为“名称”，属性值为“110kv变压器模型”，该条属性只读，该条属性在属性表可见
	GimGeBase::PEPropertyItem item(propName, propValue, true, true);
	//增加一条属性，如果在同一个属性分组中增加多条属性，可以使用addpropItems接口
	dlComp->addPropItem(item, L"基本属性");
	//增加一个名为“工程属性”的实例属性分组
	dlComp->addPropGroup(L"工程属性", false);
	propName = L"数量";
	propValue = L"1个";
	//该条属性可写，该条属性在属性表可见
	GimGeBase::PEPropertyItem itemnew(propName, propValue, false, true);
	dlComp->addPropItem(itemnew, L"工程属性");
	//更新持久化数据
	dlComp->update(project);
	GeTransform trans = GeTransform::createIdentityMatrix();
	GePoint3d point;
	point.x = -1000;
	point.y = 0;
	point.z = 0;
	trans.setTranslation(point);
	//在（-1000,0,0）的地方根据类型对象创建一个实例对象
	GimGeBase::PBDLComponentPtr insDlc = dlComp->createInstance(trans, model->GetModelId(), project);
	return;

}

//测试命令：克隆一个类型对象作为类型2，并修改该类型2对象里的一条类型数据并导入类型管理器
static void colneComponent()
{
	std::wstring path = L"PLATFORM\\BPGroup\\ComponentLib\\变电\\电气\\变压器\\三绕组变压器\\110kV变压器模型16.bfa";
	std::wstring bfaPath = BIMBase::Core::BPApplication::getInstance().getAppPath().c_str() + path;
	std::string strBfaPath = wstring2string(bfaPath);
	std::vector<Data::BPObjectPtr> importObjs;
	//先导入一个bfa模型作为类型1
	if (!GimGeBase::PEGlobalDLComponentPropManager::importBfa(strBfaPath, importObjs))
		return;
	Data::BPObjectPtr object = importObjs.at(0);
	if (!object.isValid())
		return;
	GimGeBase::PBDLComponentPtr dlcomp = dynamic_cast<GimGeBase::PBDLComponentP>(object.get());
	if (!dlcomp.isValid())
		return;
	//给类型1增加一条类型属性分组
	dlcomp->addPropGroup(L"基本属性", true);
	std::wstring propName = L"名称";
	std::wstring propValue = L"110kv变压器模型";
	GimGeBase::PEPropertyItem item(propName, propValue);
	//给类型1增加一条类型属性
	dlcomp->addPropItem(item, L"基本属性");
	dlcomp->update();
	Data::BPObjectPtr objectNew;
	BPProjectP project = BPProject::getActiveProject();
	BIMBase::Core::BPModelBaseP model = project->getActiveModel();
	std::string bfaName = "110kV变压器模型16.bfa";
	std::string typeName = "类型1";
	//根据bfa名称和type名称获得类型对象
	GimGeBase::PBDLComponentPtr dlComp = GimGeBase::PEGlobalDLComponentPropManager::getTypeByBfaAndTypeName(project, bfaName, typeName);
	if (!dlComp.isValid())
		return;
	//克隆一份类型
	dlComp = dlComp->cloneType(project);
	//对该类型修改这条类型属性
	std::wstring groupName = L"基本属性";
	propName = L"名称";
	propValue = L"220kv变压器模型";
	GimGeBase::PEPropertyItem itemnew(propName, propValue);
	dlComp->editPropertyByKey(propName, itemnew, groupName);
	//将该类型的type名称设置为类型2
	dlComp->setBfaTypeName(L"类型2");
	//向类型管理器导入一个新类型
	GimGeBase::PBDLComponentPtr objnew;
	if (!GimGeBase::PEGlobalDLComponentPropManager::importNewType(dlComp, project, objnew))
		return;

	GeTransform trans = GeTransform::createIdentityMatrix();
	GePoint3d point;
	point.x = 3000;
	point.y = 0;
	point.z = 0;
	trans.setTranslation(point);
	//在（3000,0,0）的地方根据类型对象创建一个实例对象
	GimGeBase::PBDLComponentPtr insDlc = objnew->createInstance(trans, model->GetModelId(), project);
	return;
}


AutoDoRegisterFunctionsBegin
BIMBase::BPToolsManager::registerFun("testimportcom", &testImportComponent);
BIMBase::BPToolsManager::registerFun("clonecom", &colneComponent);
AutoDoRegisterFunctionsEnd