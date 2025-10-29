#include "stdafx.h"



TransactionDemo::TransactionDemo()
{
}


TransactionDemo::~TransactionDemo()
{
}

void TransactionDemo::Transaction()
{

	BPProjectPtr project = BIMBase::Core::BPProject::getMainProject();
	if (!project.isValid())
		return;

	project->getTransactionManager().startTransactionGroup(true);
	BPTransactionPos pos = project->getTransactionManager().getCurrGroupStartPos();

	::BIMBase::PModelId modelId = project->getActiveModel()->getModelId();
	DemoObject::CubeDemo Cube;

	BPPlacement placementNew = Cube.getPlacement();
	GePoint3d ptOri = GePoint3d::create(0, 0, 0);
	placementNew.setOrigin(ptOri);

	//设置基本信息
	Cube.setHeight(3000);
	Cube.setWidth(3000);
	Cube.setLength(5000);

	if (SUCCESS != Cube.addToProject(*project, modelId))
	{
		AfxMessageBox(L"Can not add to project!");
	}

	DemoObject::OpenningDemo  opening;
	if (SUCCESS != opening.addToProject(*project, modelId))
	{
		AfxMessageBox(L"Can not add to project!");
		project->getTransactionManager().cancelToPos(pos);
	}

	project->getTransactionManager().endTransactionGroup();
}
AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("transactionDemo", TransactionDemo::Transaction);
AutoDoRegisterFunctionsEnd