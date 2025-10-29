#include "stdafx.h"
#include "createEntitysLinkDemo.h"

int g_newview = 0;
PModelId g_ModelId = 0;
createEntitysLinkDemo::createEntitysLinkDemo()
{
}


createEntitysLinkDemo::~createEntitysLinkDemo()
{
}
//多视口的联动，是在事件ElementChangeEventDemo里写的
//注意视口0布置的是3维，视口1是显示对应的2维的，如果在视口1布置3维，最后视口1显示的也是对应的2维，然后在视口0里对应显示3维
void createEntitysLinkDemo::Create()
{
	//先布置一个立方体
	BPUserInputManager::exeCommand("layoutCubeDemo");
	vector<int> vct;
	BPViewManager::getInstance().getAllActiveViewports(vct);
	bool bIsmutiviews = vct.size() == 1 ? false : true;
	//创建双视口
	if (!bIsmutiviews)
	{
		BPUserInputManager::exeCommand("view_style_OPEN_NEW");
		g_newview = BPViewManager::getInstance().getActiveIndex();
	}

	//创建新的model
	P3DStatus status;
	p3d::Utf8String sModelName = "newModel";
	p3d::platform::P3DModelType modelType = p3d::platform::P3DModelType::enPhysical;
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BIMBase::Core::ModelTreeItemInfo modelTreeItemInfo;
	BPModelPtr ptrNewModel = pProject->createNewModel(status, sModelName, modelType, true, modelTreeItemInfo);
	if (ptrNewModel.isNull())
		return ;

	g_ModelId = ptrNewModel->getModelId();

	//创建的新model在view中显示
	BPViewManager::getInstance().displayModelOnViewPort(g_ModelId, g_newview);


}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("createEntitysLinkDemo", createEntitysLinkDemo::Create);
AutoDoRegisterFunctionsEnd