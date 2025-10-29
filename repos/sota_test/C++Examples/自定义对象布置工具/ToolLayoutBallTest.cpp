#include "stdafx.h"
#include "ToolLayoutBallTest.h"
//#include "PBBimApp\PBimsOperaterErrHelp.h"

using namespace TestObject;

ToolLayoutBallTest::ToolLayoutBallTest()
{
	m_ptrCube = CubeTest::create();
}


ToolLayoutBallTest::~ToolLayoutBallTest()
{

}

void ToolLayoutBallTest::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
	//PBMessageCenter::OutPutPrompt(L"请选择点");
}

void ToolLayoutBallTest::_onRestartTool()
{
	ToolLayoutBallTest* newTool = new ToolLayoutBallTest();
	newTool->installTool();
}

bool ToolLayoutBallTest::_onDataButton(BPBaseButtonEventCP ev)
{
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (pViewPort == nullptr)
		return false;
	int nViewIndex = pViewPort->GetViewNumber();
	BPViewportP pVP = BPViewManager::getInstance().getViewport(nViewIndex);
	if (NULL == pVP) return false;

	BIMBase::Core::BPProjectP pProject = ev->getViewport()->GetWriteModel()->getBPProject();
	if (pProject == nullptr)
		return false;
	GePoint3d ptSel = *ev->getPoint();

	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	if (pModel == nullptr)
		return false;

	GePoint3d ptE = *ev->getPoint();
	ptrBallTest ptrBall = BallTest::create();
	if (ptrBall.isNull())
		return false;
	ptrBall->setOrigin(ptE);
	//增加构件到工程中
	if (SUCCESS != ptrBall->addToProject(*project, pModel->getModelId()))
	{
		AfxMessageBox(L"Can not add to project!");
		return false;
	}

	if (m_ptrCube.isNull())
		return true;

	if (!m_ptrCube->getDataKey().isValid())
		return true;

	//与Cube建立关联关系
	BPDataId relationId;
	BPRelationshipInserter::addRelationship(&relationId, *project, 
		                                          *m_ptrCube->getData(*project), *ptrBall->getData(*project), 
		                                          PBM_SCHEMA_TEST, PBM_RELSHIP_CUBEWITHBALL);

	return true;
}


bool ToolLayoutBallTest::_onResetButton(BPBaseButtonEventCP)
{
	_exitTool();
	return true;
}

void ToolLayoutBallTest::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->GetViewport()->GetTargetModel()->getBPProject();
	::BIMBase::PModelId curModelId = ev->GetViewport()->GetTargetModel()->GetModelId();

	int n = 0;
	p3d::PString reasonDesc = L"";
	BIMBase::Core::BPPickDataPtr path = BPEntityLocateManager::getInstance().doLocatePickData(n, /*NULL*/reasonDesc, true, ev->getPoint(), ev->getViewport());
	if (path == nullptr)
		return;

	BPEntityP pElement = path->getEntity();
	if (pElement == nullptr)
		return;

	if (pElement->GetElementRef()->getClassId() != BIMBase::Core::BPDataUtil::GetClassIdByClassName(*pProject, PBM_SCHEMA_TEST, PBM_CLASS_CUBE_TEST))
		return;

	m_ptrCube->initFromData(*BPDataUtil::getDataOnEntity(*pElement));

	ptrBallTest ptrBall = BallTest::create();
	ptrBall->setOrigin(*ev->getPoint());
	m_ptrGraphic = ptrBall->createPhysicalGraphics(*pProject, curModelId, true);

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(m_ptrGraphic->getEntityR());
}

bool ToolLayoutBallTest::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

//对工具进行注册
BPTool* CreateBallTool()
{
	ToolLayoutBallTest* tool = new ToolLayoutBallTest();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("LayoutBall", &CreateBallTool);
AutoDoRegisterFunctionsEnd