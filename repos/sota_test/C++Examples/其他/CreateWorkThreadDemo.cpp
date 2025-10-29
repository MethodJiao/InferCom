#include "stdafx.h"
using namespace BIMBase::ParaComponent;

void CreateWorkThreadDemo()
{
	string bfaPath = "D:\\房子.bfa";
	auto placeBfaLambda = [](void* file)
	{
		BIMBase::FrameWork::ProgressCtrlTypeInfo info;
		BIMBase::FrameWork::BPProgressBall noteProxy(info);
		std::wstring note = _T("开始布置");
		noteProxy.setPBProgressStatusText(note);
		noteProxy.setPBProgressPos(50);
		BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
		if (pProject == nullptr)
			return;
		BPModelPtr ptrModel = BPModel::getActiveModel();
		if (!ptrModel)
			return;
		string bfaPaths = *(string*)file;
		BIMBase::ParaComponent::BPComponentBfaHandle bfaHandle = BIMBase::ParaComponent::BPComponentBfaManager::addImported(pProject, bfaPaths);
		if (!bfaHandle.valid())
			return;
		std::vector<std::string> listTypeName = bfaHandle.getTypeNameList();
		if (listTypeName.size() == 0)
			return;
		BPComponentBfacomponentEditer editer = bfaHandle.createBfacomponentEditer(listTypeName.at(0));
		BPComponentBfaPlacedEditHandle placeEditHandle1 = BPComponentBfaManager::placeObject(pProject, editer.getObject(), ptrModel.get());
		if (!placeEditHandle1.valid())
		{
			Utf8String utf8 = "";
			P3DStringHelper::wCharToUtf8(utf8,L"布置失败");
			BIMBase::FrameWork::BPAsynchronousUtil::sendMessageToUI("P3D_BCG_Messagebox", utf8.c_str());
			return;
		}
		noteProxy.setPBProgressPos(100);

	};
	BIMBase::FrameWork::BPAsynchronousUtil::sendMessageToWork(placeBfaLambda,&bfaPath,false,true);
	return;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("CreateWorkThreadDemo", &CreateWorkThreadDemo);
AutoDoRegisterFunctionsEnd