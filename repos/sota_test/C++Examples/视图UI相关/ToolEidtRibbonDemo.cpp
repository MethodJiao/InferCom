#include "stdafx.h"
#include "ToolEidtRibbonDemo.h"
using namespace BIMBase::FrameWork;

ToolEidtRibbonDemo::ToolEidtRibbonDemo()
{

}

ToolEidtRibbonDemo::~ToolEidtRibbonDemo()
{

}

void ToolEidtRibbonDemo::addRibbon()
{
	std::vector<CString> vctStr;
	BPRibbonUtil::ribbonGetAllCategoryName(vctStr);
	int nNum = vctStr.size();

	//增加Tab页
	BPRibbonUtil::ribbonAddCategory(_T("动态测试"), NULL, NULL, CSize(16, 16), CSize(32, 32), nNum);

	//增加panel
	BPRibbonUtil::ribbonAddPanel(_T("动态测试"), _T("动态面板"));

	HICON hIcon = AfxGetApp()->LoadIconW(IDI_Demo_ICON);

	//增加button,
	BPRibbonUtil::ribbonAddButton(_T("动态测试"), _T("动态面板"), _T("大图标"), _T("LayoutCubeDemo"), hIcon);

	//刷新菜单栏
	BPRibbonUtil::ribbonRecalcLayout();
}

void ToolEidtRibbonDemo::RemoveRibbon()
{
	std::vector<CString> vctStr;
	BPRibbonUtil::ribbonGetAllCategoryName(vctStr);
	int nNum = vctStr.size();

	//增加Tab页
	BPRibbonUtil::ribbonRemoveCategory(_T("动态测试"));

	//增加panel
	BPRibbonUtil::ribbonRemovePanel(_T("动态测试"), _T("动态面板"));

	//增加button,
	BPRibbonUtil::ribbonRemoveButton(_T("动态测试"), _T("动态面板"), _T("大图标"));

	//刷新菜单栏
	BPRibbonUtil::ribbonRecalcLayout();
}

void ToolEidtRibbonDemo::BlackRibbon()
{

	//添加黑名单
	BPRibbonUtil::addCategoryToBlackList(L"CPP范例1", 1003, L"RibbonDesignCPPDemo.dll");

	//打开黑名单
	BPRibbonUtil::setOpenBlackList(true, 1003, L"RibbonDesignCPPDemo.dll");

	//刷新菜单栏
	BPRibbonUtil::ribbonRecalcLayout();
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("addRibbonDemo"), &ToolEidtRibbonDemo::addRibbon);
BPToolsManager::registerFun(_T("RemoveRibbonDemo"), &ToolEidtRibbonDemo::RemoveRibbon);
BPToolsManager::registerFun(_T("BlackRibbonDemo"), &ToolEidtRibbonDemo::BlackRibbon);
AutoDoRegisterFunctionsEnd