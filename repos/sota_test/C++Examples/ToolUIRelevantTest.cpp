#include "stdafx.h"
#include "ToolUIRelevantTest.h"
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
}BEGIN_MESSAGE_MAP(ExternPaneDlg, CDialogEx)
ON_WM_CONTEXTMENU()
END_MESSAGE_MAP()


ToolUIRelevantTest::ToolUIRelevantTest()
{
}


ToolUIRelevantTest::~ToolUIRelevantTest()
{
}

BPNewDockContainer* myDock1 = nullptr;
void ToolUIRelevantTest::initDockTest()
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

void ToolUIRelevantTest::rButtonClickTest()
{
	////右键菜单
	////空白视图点击右键
	//BPRButtonClickEvent::getInstance().setMenuCommand(std::make_pair(_T("空白右键test"), _T("LayoutLine")));
	////选中构件点击右键
	//BPRButtonClickSelectionEvent::getInstance().setMenuCommand(std::make_pair(_T("选中test"), _T("LayoutSolid")));

}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("ToolDockTest", &ToolUIRelevantTest::initDockTest);
BPToolsManager::registerFun("ToolRButtonClickTest", &ToolUIRelevantTest::rButtonClickTest);
AutoDoRegisterFunctionsEnd