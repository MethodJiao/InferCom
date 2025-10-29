// Examples.cpp : 定义 DLL 的初始化例程
#include "stdafx.h"
#include "ExampleDef.h"
#include "Examples.h"
#include "ProjectEventDemo.h"
#include "ArchWallDragManipulatorDemo.h"
#include "DomainChangeEventDemo.h"
#include "EntitySymbologyEventDemo.h"
#include "ToolClashDemo.h"
#include "ObjectChangeEventDemo.h"
#include "adsarxfunc/ADSARXFUN.h"
#include "MyCubeDragManipulatorDemo.h"
#include "CubeDemo.h"
#include "MyUBDragManipulatorDemo.h"
#include "ElementChangeEventDemo.h"
#include "ObjectChangeEventDemo.h"
#include "BPUIFrameWork/BPToolBarUtil.h"


#pragma comment (lib, "adsarxfunc.lib")

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

PBBIM_IMPLEMENT_EXTENSION_MODULE(ExampledllDemo)

static CDialog* NewSingleDockDlg()
{
	return nullptr;
}

class ExampleMainFrameEventListener : public BIMBase::BPMainFrameEventListener
{
protected:
	virtual void    _onPostCreate(HWND& mainFrame) override
	{
		PBBimModuleResourceOverride resOverride;
	}
};
static ExampleMainFrameEventListener s_mainFrameListener;

using namespace BIMBase::FrameWork;
//////增加工具条注意事项
//1、为工具条配置的图片一定要是16px的PNG图片，图片路径为Support\ToolBarPicture\LightPic（浅色模式）下，
// 深色模式在Support\ToolBarPicture\DarkPic路径下
//2、工具条增加需要在插件加载时执行
static void addToolBarButton()
{
	//创建名为 测试工具条 的 二次开发专业工具条
	int toolIndex_1st = BPToolBarUtil::createToolBar(L"测试工具条", L"二次开发CPP");
	CString ButtonTips;
	CString Commandstr;

	//初始化第一个按钮
	BPToolBarUtil::BPToolBarButtonInfo info;
	info.PicName = L"DemoToolBarPic\\sphere16.png";
	info.ButtonTips = L"线框模式";
	info.Commandstr = L"view_style_WIREFRAME";
	info.IsSeparator = false;
	//初始化一个分隔符
	BPToolBarUtil::BPToolBarButtonInfo info1;
	info1.IsSeparator = true;
	//初始化第二个按钮
	BPToolBarUtil::BPToolBarButtonInfo info2;
	info2.PicName = L"DemoToolBarPic\\sphere16.png";
	info2.ButtonTips = L"隐藏线模式";
	info2.Commandstr = L"view_style_HIDDEN_LINE";
	info2.IsSeparator = false;
	//初始化第三个按钮
	BPToolBarUtil::BPToolBarButtonInfo info3;
	info3.PicName = L"DemoToolBarPic\\model16.png";
	info3.ButtonTips = L"";
	info3.Commandstr = L"";
	info3.IsSeparator = false;

	//初始化第四个按钮 为combobox
	BPToolBarUtil::BPToolBarComboBoxButtonInfo cbinfo;
	cbinfo.buttonWidth = 75;
	cbinfo.Itemnum = 3;
	cbinfo.selectedIndex = 2;
	cbinfo.vecItemstr.push_back(L"第一条");
	cbinfo.vecItemstr.push_back(L"第二条");
	cbinfo.vecItemstr.push_back(L"第三条");

	cbinfo.vecCommandstr.push_back(L"$PBbim.MultiView.OneView");
	cbinfo.vecCommandstr.push_back(L"$PBbim.MultiView.TwoView");
	cbinfo.vecCommandstr.push_back(L"$PBbim.MultiView.ThreeView");

	//依次调用addbutton初始化工具条
	BPToolBarUtil::addButton(toolIndex_1st, L"二次开发CPP", info);
	BPToolBarUtil::addButton(toolIndex_1st, L"二次开发CPP", info1);
	BPToolBarUtil::addButton(toolIndex_1st, L"二次开发CPP", info2);
	int button3 = BPToolBarUtil::addButton(toolIndex_1st, L"二次开发CPP", info3);
	BPToolBarUtil::addComboBoxButton(L"二次开发CPP", toolIndex_1st, cbinfo);
	//将第三个按钮设置为常亮状态
	BPToolBarUtil::setButtonCheck(button3, TRUE);

	//创建名为 测试工具条2 的 二次开发专业工具条
	int toolIndex_2nd = BPToolBarUtil::createToolBar(L"测试工具条2", L"二次开发CPP");
	{
		BPToolBarUtil::BPToolBarButtonInfo info;
		info.PicName = L"DemoToolBarPic\\dummy16.png";
		info.ButtonTips = L"亮色模式";
		info.Commandstr = L"$P3D.LightStyle";
		info.IsSeparator = false;

		BPToolBarUtil::BPToolBarButtonInfo info1;
		info1.IsSeparator = true;

		BPToolBarUtil::BPToolBarButtonInfo info2;
		info2.PicName = L"DemoToolBarPic\\cone16.png";
		info2.ButtonTips = L"深色模式";
		info2.Commandstr = L"$P3D.DarkStyle";
		info2.IsSeparator = false;

		BPToolBarUtil::BPToolBarButtonInfo info3;
		info3.IsSeparator = true;


		UINT buttonenble = BPToolBarUtil::addButton(toolIndex_2nd, L"二次开发CPP", info);
		BPToolBarUtil::addButton(toolIndex_2nd, L"二次开发CPP", info1);
		BPToolBarUtil::addButton(toolIndex_2nd, L"二次开发CPP", info2);
		BPToolBarUtil::addButton(toolIndex_2nd, L"二次开发CPP", info3);
		//将第一个按钮设置为禁用
		BPToolBarUtil::setButtonEnable(buttonenble, false);

		//创建名为 测试工具条3 的 二次开发专业工具条
		//其中放置宽度为180px文本居中显示的文本框按钮
		int toolIndex_3nd = BPToolBarUtil::createToolBar(L"测试工具条3", L"二次开发CPP");
		{
			BPToolBarUtil::BPToolBarTextButtonInfo cbinfo;
			cbinfo.buttonWidth = 180;
			cbinfo.content = L"我的宽度是180px";
			cbinfo.textAlignment = 1;
			BPToolBarUtil::addTextButton(L"二次开发CPP", toolIndex_3nd, cbinfo);
		}
	}
}

static ProjectEventDemo s_projectEventDemo;
static DomainChangeEventDemo s_domainchangeEventDemo;
static ObjectChangeEventDemo s_ObjectChangeEventDemo;
static RButtonClickEventDemo s_RButtonClickEventDemo;
static ElementChangeEventDemo s_ElementChangeEventDemo;
using namespace DemoObject;

#include "BPPluginManager/BPPluginManagerApi.h"
class BPPluginDemo :public Plugin::BPPluginFactory
{
public:
	BPPluginDemo() {};
	virtual ~BPPluginDemo() {};

protected:
	virtual void _onLoadPlugin(CWnd* curMainFrame) {};
	virtual void _onUnLoadPlugin(::BIMBase::Core::BPProjectR curProject) {};
	virtual void _initPluginInformation(Plugin::PluginInformation& pluginInfomation)
	{
		return;
	}
};
static BPPluginDemo _pluginDemo;

extern "C" int APIENTRY

DllMain(HINSTANCE hInstance, DWORD dwReason, LPVOID lpReserved)
{
	// 如果使用 lpReserved，请将此移除
	UNREFERENCED_PARAMETER(lpReserved);

	if (dwReason == DLL_PROCESS_ATTACH)
	{
		addToolBarButton();
		TRACE0("ExampledllDemo.DLL 正在初始化!\n");

		BIMBase::Plugin::BPPluginManager::getInstance().registerPlugin(&_pluginDemo);
		BPMainFrameEventListenerCenter::getInstance().addListener(&s_mainFrameListener);
		BPProjectEventListenerCenter::getInstance().addListener(&s_projectEventDemo);
		BPDomainEventHandleManager::getInstance()->registerDomainEventHandler(BPDomainEnvironment::getInstance()->getDomainCodeByKeyName(L"二次开发CPP"), &s_domainchangeEventDemo);
		BPObjectChangeEventListenerCenter::getInstance().addListener(&s_ObjectChangeEventDemo);
		BIMBase::FrameWork::BPViewRButtonClickListenerCenter::getInstance().addListener(BPDomainEnvironment::getInstance()->getDomainCodeByKeyName(L"二次开发CPP"), &s_RButtonClickEventDemo);
		BPEntityChangeEventListenerCenter::getInstance().addHandler(&s_ElementChangeEventDemo);
		//注册造型夹点
		std::wstring classname = std::wstring(L"DemoSchema").append(L":").append(L"CubeDemo");
		std::wstring classnameUB = std::wstring(PBM_SCHEMA_Demo_W).append(L":").append(PBM_CLASS_UNIVERSAL_BEAM_Demo_W);
		BPManipulatorExtensionManager::registerDragManipulatorExtension(classname.c_str(), new MyCubeDragManipulatorDemoExtension());
		BPManipulatorExtensionManager::registerDragManipulatorExtension(classnameUB.c_str(), new MyUBDragManipulatorDemoExtension());
		//注册运行时类型判断

		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, PBM_CLASS_CUBE_Demo, new CubeDemoExtension());
		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, "DrainDemo", new DrainDemoExtension());
		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, "EmbankmentDemo", new EmbankmentDemoExtension());
		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, PBM_CLASS_OPENNING_Demo, new OpenningDemoExtension());
		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, PBM_CLASS_INDEPENDENT_BRIDGE, new IndependentBridgeExtension());
		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, PBM_CLASS_UNIVERSAL_BEAM_Demo, new UniversalBeamDemoExtension());
		BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo,
			PBM_CLASS_UNIVERSAL_BEAM_Demo, new UniversalBeamDemoExtension());

		BIMBase::BPUserInputManager::exeCommand(L"AddRibbonDemo");
		if (!ExampledllDemo.AttachInstance(hInstance))
			return 0;
	}
	else if (dwReason == DLL_PROCESS_DETACH)
	{
		TRACE0("ExampledllDemo.DLL 正在终止!\n");
		BIMBase::BPUserInputManager::exeCommand(L"RemoveRibbonDemo");
		BIMBase::Plugin::BPPluginManager::getInstance().unRegisterPlugin();

		// 在调用析构函数之前终止该库
		ExampledllDemo.DetachInstance();
	}
	return 1;   // 确定
}

void AdsLine()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;
	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();

	double pFirst[3], pSecond[3];
	getPoint(nullptr, "Select first point", pFirst);
	getPoint(nullptr, "Select second point", pSecond);

	GePoint3d ptFirst = GePoint3d::create(pFirst[0], pFirst[1], pFirst[2]);
	GePoint3d ptSecond = GePoint3d::create(pSecond[0], pSecond[1], pSecond[2]);
	ptrGraphic->addGeCurve(*IGeCurveBase::createSegment(GeSegment3d::create(ptFirst, ptSecond)));
	ptrGraphic->save();
}


void   addGeCurve()
{

	GeSegment3d seg;
	seg.point[0].set(0, 0, 0);
	seg.point[1].set(1000, 1000, 1000);

	GeSegment3d seg1;
	seg1.point[0].set(0, 0, 0);
	seg1.point[1].set(3000, 8000, 8000);

	IGeCurveBasePtr line = IGeCurveBase::createSegment(seg);
	IGeCurveBasePtr line1 = IGeCurveBase::createSegment(seg1);

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;
	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
	BIMBase::BPSymbology type;
	type.color = 10;
	type.style = 2;
	type.weight = 5;

	BIMBase::BPSymbology type2;
	type2.color = 5;
	type2.style = 1;
	type2.weight = 3;
	ptrGraphic->addGeCurve(*line, type);
	ptrGraphic->addGeCurve(*line1, type2);
	ptrGraphic->save();
}
void getdatafrompythonscrip(const std::string & scriptpath)
{
	std::string path = scriptpath;
	for (auto& cpy : path)
	{
		if (cpy == '\\')
			cpy = '/';
	}
	BIMBase::Core::Noumenon noumenon = BIMBase::Core::PythonFunction::getNoumenonFromScript(path);
	BIMBase::Core::BPParametricComponentRepresentation rep;
	Data::BPObjectPtr object = rep.createObject(noumenon);
}
AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("adsLineDemo"), &AdsLine);
BPToolsManager::registerFun(_T("addGeCurve"), &addGeCurve);
AutoDoRegisterFunctionsEnd