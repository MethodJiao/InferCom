#include "stdafx.h"
#include "ToolTestDockDlg.h"
#include "Resource.h"


//using namespace PBBim::PBBimCore;

//#define KEY_CUSTOMPROPERTY  L"CustomProperty"


ToolTestDockDlg::ToolTestDockDlg()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	//TMY暂时
	//BP_MESSAGE_DOCKDLG_OPEN("TestDockDlg");
}


ToolTestDockDlg::~ToolTestDockDlg()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->Clear();
}


void ToolTestDockDlg::_onPostInstall()
{
	//调用基类

	//BP_MESSAGE_DOCKDLG_OPEN("TestDockDlg");
	T_Super::_onPostInstall();
	//_buildAgenda(NULL);
	//ElemSource es = _getElemSource();
	_setLocateCursor(true);
	BPSnap::getInstance().enableSnap(true);

}


void ToolTestDockDlg::_onRestartTool()
{
	//重启工具
	ToolTestDockDlg* newTool = new ToolTestDockDlg();
	newTool->installTool();
}


void ToolTestDockDlg::_setupAndPromptForNextAction()
{
	/*if (SOURCE_Pick != _getElemSource())
		return;*/
}


bool ToolTestDockDlg::_onDataButton(BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return false;
	BPProjectPtr ptrProject = pProjectManager->getMainProject();
	if (ptrProject.isNull())
		return false;


	//size_t nSelect = SelectionSetManager::GetManager().NumSelected();
	size_t nSelect = m_vcEEH.size();

	if (nSelect < 1)
	{
		return true;
	}

	short elemType = -1;

	for (size_t i = 0; i < nSelect; i++)
	{
		BIMBase::Core::BPEntityP curSel = m_vcEEH[i]->GetElementRef();
		if (curSel == nullptr)
			continue;

		BIMBase::Data::IBPObjectPtr ptrCopy = BPObjectExtensionManager::getInstance().getBPObject(*curSel);
		if (!ptrCopy.isValid())
			continue;

		Utf8String className = ptrCopy->getClassName();

	}

	return true;
}


bool ToolTestDockDlg::_onResetButton(BPBaseButtonEventCP ev)
{
	//点击右键退出工具
	//_ExitTool();

	getEntityArray()->clear();
	BPSelectionSetManager::getInstance().emptyAll();
	
	//获取当前视图，强制刷新界面

	//IndexedViewportP viewPort = IViewManager::GetManager().getActivedViewport();
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();

	int nViewIndex = pViewPort->getViewNumber();
	BPViewportP pVp = BPViewManager::getInstance().getViewport(nViewIndex);
	if (NULL == pVp) return false;
	if (pVp)
	{
		pVp->updateView();
		/*FullUpdateInfo info;
		vp->UpdateView(info);*/
	}


	return true;
}



/*
void  ToolTestDockDlg::_OnReceive(Utf8CP messageType, JsonValueCR messageDataObj)
{
	TOOLSETTING_CHECK;
}*/



BPEntityPtr ToolTestDockDlg::_buildLocateAgenda(BPPickDataCP path, BPBaseButtonEventCP ev)
{
	BPEntityPtr ptrEh = T_Super::_buildLocateAgenda(path, ev);
	//     PDEditEntityHandle eeh;
	//     eeh.Duplicate(*P3DInnerFriend::getInnerEntityP(*eh->GetElementRef()));
	m_vcEEH.push_back(ptrEh);
	return ptrEh;
}




/*
StatusInt ToolTestDockDlg::_onEntityModify(BPEntityR el)
{
	return ERROR;
}*/


bool ToolTestDockDlg::_onModifierKeyTransition(bool wentDown, int key)
{
	return __super::_onModifierKeyTransition(wentDown, key);
}


void ToolTestDockDlg::_exitTool()
{
//TMY暂时
	//BP_MESSAGE_DOCKDLG_CLOSE("TestDockDlg");
	__super::_exitTool();
}



//对工具进行注册
BPTool* CreateToolTestDockDlg()
{
	ToolTestDockDlg* tool = new ToolTestDockDlg();
	return tool;
}




AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("TestDockDlg", &CreateToolTestDockDlg);
AutoDoRegisterFunctionsEnd