#include "stdafx.h"
#include "DockDlgDemo.h"
using namespace BIMBase::FrameWork;


IMPLEMENT_DYNAMIC(ExternPaneDlg, CDialogEx)
ExternPaneDlg::ExternPaneDlg()
{
}

ExternPaneDlg::~ExternPaneDlg()
{
}

BOOL ExternPaneDlg::OnInitDialog()
{
	return TRUE;
}

BEGIN_MESSAGE_MAP(ExternPaneDlg, CDialogEx)
ON_WM_CONTEXTMENU()
END_MESSAGE_MAP()


//直接通过命令显示停靠对话框
BPNewDockContainer* myDock1 = nullptr;
static void initDockDemo()
{
	PBBimModuleResourceOverride resOverride;
	// 停靠窗口
	ExternPaneDlg * dlg1 = new ExternPaneDlg();
	dlg1->Create(IDD_DIALOG_Dock, BPNewDockManager::getInstance().getDock(0));
	myDock1 = BPNewDockManager::getInstance().getDock(0);
	if (myDock1 != NULL)
	{
		myDock1->SetDlg(dlg1);
		myDock1->SetDockPosition(L"right");
		myDock1->ShowDockContainer(true);
		myDock1->SetWindowTextW(L"123");
	}
}

//用于右键菜单显示或隐藏停靠对话框
static void showOrHideDockDlg()
{
	if (myDock1 != nullptr)
	{
		myDock1->ShowControlBar(!myDock1->IsVisible(), FALSE, TRUE);
	}
	else
	{
		PBBimModuleResourceOverride resOverride;
		// 停靠窗口
		ExternPaneDlg* dlg1 = new ExternPaneDlg();
		dlg1->Create(IDD_DIALOG_Dock, BPNewDockManager::getInstance().getDock(0));
		myDock1 = BPNewDockManager::getInstance().getDock(0);
		if (myDock1 != NULL)
		{
			myDock1->SetDlg(dlg1);
			myDock1->SetDockPosition(L"right");
			myDock1->ShowDockContainer(true);
			myDock1->SetWindowTextW(L"123");
		}
	}
}
//用于右键菜单显示或隐藏停靠对话框
static bool getShowOrHideDockDlg()
{
	if (myDock1 != nullptr)
	{
		BOOL bIsVisiable = myDock1->IsVisible();
		return bIsVisiable;
	}
	return false;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("toolDockDemo", &initDockDemo);
BPToolsManager::registerFun(L"DemoShowOrHideDockDlgDemo", &showOrHideDockDlg);
DependencyInversion::instance("getShowOrHideDockDlgDemo").set(getShowOrHideDockDlg);
AutoDoRegisterFunctionsEnd