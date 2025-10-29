#include "stdafx.h"
#include "ProjectTreeDemoNew.h"

using namespace BIMBase::ProjectTree;

//项目树注册
REGISTERPJTCLASS(ProjectTreeDemoNew)

#define IDM_CREATE_NODE  12001

static int g_nTreeNodeCount = 1;
void ProjectTreeDemoNew::_preAdd(IN BIMBase::Data::BPTreeNodeP pNode)
{
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectP project = pProjectManager->getMainProject();
	if (project == NULL)
		return;

	P3DStatus status;
	p3d::Utf8String modelName = "";
	modelName.sprintf("newModel%d", g_nTreeNodeCount);

	g_nTreeNodeCount++;
	p3d::platform::P3DModelType modelType = p3d::platform::P3DModelType::enPhysical;
	BIMBase::Core::ModelTreeItemInfo modelTreeItemInfo;
	BPModelPtr newModel = project->createNewModel(status, modelName, modelType, true, modelTreeItemInfo);
	if (newModel.isNull())
		return;


	Int32 nModelId = newModel->getModelId().m_id;

	std::wstring sId = std::to_wstring(nModelId);

	vector<std::wstring> resource;
	resource.push_back(sId);
	for (auto res : resource)
	{
		pNode->attachResource(res);
	}

	pNode->updateInProject(*project);

	BPViewManagerR mg = BPViewManager::getInstance();
	UInt32 nIndex = mg.getActiveIndex();

	//显示创建的model
	BPViewManager::getInstance().displayModelOnViewPort(nModelId, nIndex);
}

void ProjectTreeDemoNew::_posAdd(IN BIMBase::Data::BPTreeNodeP pNode, IN bool bSelect)
{
	int a = 0;
}

void ProjectTreeDemoNew::_preDelete(IN BIMBase::Data::BPTreeNodeP pNode)
{

}

void ProjectTreeDemoNew::_onDbClick(IN BIMBase::Data::BPTreeNodeP pSelectNode)
{
	if (pSelectNode == nullptr)
		return;

	std::vector<std::wstring> vctStr = pSelectNode->getResourceVec();
	if (vctStr.size() == 0)
		return;

	Int32 Id = _wtoi(vctStr[0].c_str());

	BPViewManagerR mg = BPViewManager::getInstance();
	UInt32 nIndex = mg.getActiveIndex();

	BPViewManager::getInstance().displayModelOnViewPort(Id, nIndex);
}



void ProjectTreeDemoNew::_onRightClick(IN BIMBase::Data::BPTreeNodeP pSelectNode)
{
	if (pSelectNode == nullptr)
		return;

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectP project = pProjectManager->getMainProject();
	if (project == NULL)
		return;

	CPoint cp;
	GetCursorPos(&cp);

	CMenu pMenu;
	VERIFY(pMenu.CreatePopupMenu());

	pMenu.AppendMenuW(MF_STRING, IDM_CREATE_NODE, _T("创建节点"));

	CWnd* cwnd = AfxGetMainWnd();
	int retID = pMenu.TrackPopupMenu(TPM_LEFTALIGN | TPM_RETURNCMD | TPM_NONOTIFY, cp.x, cp.y, cwnd);

	if (retID == IDM_CREATE_NODE)
	{
		BPTreeNodeParam nodeParam;
		nodeParam.nodeType = BPTreeNodeType::enScene;
		std::wstring sName = std::to_wstring(g_nTreeNodeCount);
		sName = _T("场景") + sName;
		nodeParam.wsNodeName = sName;
		nodeParam.bSelect = true;
		addTreeNode(pSelectNode, nodeParam);
	}
}