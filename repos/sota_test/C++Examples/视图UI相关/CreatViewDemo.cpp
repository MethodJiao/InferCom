#include "stdafx.h"
#include "CreatViewDemo.h"


void CreatViewdemo::CreateViewDemo()
{
	//获取当前工程
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == nullptr)
		return;

	//创建view
	CFrameWnd* pView = BIMBase::FrameWork::BPUIFrameWorkUtil::createView();
	if (pView == nullptr)
		return;

	//获取创建view的number
	int nNum = BIMBase::FrameWork::BPUIFrameWorkUtil::getViewNumber(pView);

	//创建新的model
	P3DStatus status;
	p3d::Utf8String modelName = "newModel";
	p3d::platform::P3DModelType modelType = p3d::platform::P3DModelType::enPhysical;
	BIMBase::Core::ModelTreeItemInfo modelTreeItemInfo;
	BPModelPtr ptrNewModel = pProject->createNewModel(status, modelName, modelType, true, modelTreeItemInfo);
	if (ptrNewModel.isNull())
		return;

	PModelId modelId = ptrNewModel->getModelId();

	//创建的新model在view中显示
	BPViewManager::getInstance().displayModelOnViewPort(modelId, nNum);
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("createViewDemo", &CreatViewdemo::CreateViewDemo);
AutoDoRegisterFunctionsEnd