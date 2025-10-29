#include "stdafx.h"
#include "DomainChangeEventDemo.h"
#include "resource.h"

using namespace DemoObject;
using namespace BIMBase::ProjectTree;

bool DomainChangeEventDemo::closedDomainNotify(::BIMBase::Core::BPProjectP pPrj)
{
	return true;
}
bool DomainChangeEventDemo::initalDomainData(::BIMBase::Core::BPProjectP pPrj)
{
	CString str = BPDomainEnvironment::getInstance()->getCurrentDomainKeyName();
	CString text = L"现在专业是" + str;
	AfxMessageBox(text);

	return true;

	str = _T("切换专业测试");
	if (BIMBase::FrameWork::BPRibbonUtil::ribbonIsExistCategory(str))
		return false;
	BIMBase::FrameWork::BPRibbonUtil::ribbonAddCategory(str, NULL, NULL, CSize(16, 16), CSize(32, 32), 0, NULL);
	BIMBase::FrameWork::BPRibbonUtil::ribbonAddPanel(str, _T("测试"), 0, NULL, TRUE, 0);
	return true;
}
bool DomainChangeEventDemo::refreshDomainUi(::BIMBase::Core::BPProjectP pPrj)
{
	CString str = BPDomainEnvironment::getInstance()->getCurrentDomainKeyName();

	BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return false;
	//使用新项目树体系创建
	BIMBase::ProjectTree::BPProjectTree* projectTree = BPProjectTreeManager::getInstance().createNewProjectTree(pProject, L"项目树", L"ProjectTreeDemoNew");
	if (str == L"二次开发CPP" || str == L"二次开发CSharp")
		BPProjectTreeManager::getInstance().setAvticeProjectTree(projectTree, true);


	return true;
}

bool DomainChangeEventDemo::refreshDomainState(::BIMBase::Core::BPProjectP pPrj)
{
	return true;
}