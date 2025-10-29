#include "stdafx.h"
#include "ToolLayoutOpenningDemo.h"

using namespace DemoObject;

ToolLayoutOpenningDemo::ToolLayoutOpenningDemo()
{
	m_ptrCube = CubeDemo::create();
	m_ptrOpenning = OpenningDemo::create();
}


ToolLayoutOpenningDemo::~ToolLayoutOpenningDemo()
{

}

void ToolLayoutOpenningDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPSnap::getInstance().enableLocate(false);
	BPSnap::getInstance().enableSnap(true);
}

void ToolLayoutOpenningDemo::_onRestartTool()
{
	ToolLayoutOpenningDemo* newTool = new ToolLayoutOpenningDemo();
	newTool->installTool();
}

bool ToolLayoutOpenningDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	BIMBase::Core::BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == nullptr)
		return false;

	BPModelP pModel = ev->getViewport()->getTargetModel();
	if (pModel == nullptr)
		return false;

	if (m_ptrCube.isNull())
		return false;

	if (!m_ptrCube->getDataKey().isValid())
		return false;

	if (m_ptrOpenning.isNull())
		return false;

	//增加构件到工程中
	if (SUCCESS != m_ptrOpenning->addToProject(*pProject, pModel->getModelId()))
	{
		AfxMessageBox(L"Can not add to project!");
		return false;
	}	

	//与Cube建立关联关系
	BPDataId relationId;
	BPRelationshipInserter::addRelationship(&relationId, *pProject,
		                                          *m_ptrCube->getData(*pProject), *m_ptrOpenning->getData(*pProject),
		                                          PBM_SCHEMA_Demo, PBM_RELSHIP_CUBEWITHOPENNING);
	m_ptrCube->replaceGraphics(*pProject);

	//获取当前视图，强制刷新界面
	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort) return false;
	if (pViewPort)
	{
		pViewPort->updateView();
	}
	return true;
}


bool ToolLayoutOpenningDemo::_onResetButton(BPBaseButtonEventCP)
{
	_exitTool();
	return true;
}

void ToolLayoutOpenningDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	m_ptrCube == nullptr;
	m_ptrOpenning = OpenningDemo::create();
	if (NULL == ev)
		return;

	//获取点击点所在的工程和模型ID
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == nullptr)
		return;
	BPModelP pModel = ev->getViewport()->getTargetModel();
	if (pModel == nullptr)
		return;

	//获取鼠标当前拾取到的entity
	int nErrorCode = 0;
	p3d::PString sReasonDesc = L"";
	BIMBase::Core::BPPickDataPtr path = BPEntityLocateManager::getInstance().doLocatePickData(nErrorCode, sReasonDesc, true, ev->getPoint(), ev->getViewport());
	if (path == nullptr)
		return;
	BPEntityP pElement = path->getEntity();
	if (pElement == nullptr)
		return;
	if (pElement->getClassId() != BIMBase::Core::BPSchemaManager::getClassIdByClassName(*pProject, PBM_SCHEMA_Demo, PBM_CLASS_CUBE_Demo))
		return;

	//将entity转为cube对象
	BPDataPtr ptrCubeData = BPDataUtil::getDataOnEntity(*pElement);
	if (ptrCubeData == nullptr)
		return;
	m_ptrCube = CubeDemo::create(*ptrCubeData);
	if (m_ptrCube == nullptr)
		return;

	//获取Cube对象端点,将鼠标点投影到cube基线上
	GeSegment3d segment = GeSegment3d::create(m_ptrCube->getStartPoint(), m_ptrCube->getEndPoint());
	GePoint3d projectPoint = GePoint3d::createByZero();
	double dPara = 0;
	segment.projectPointToSegment(projectPoint, dPara, *ev->getPoint());

	//获取cube对象的方向
	BPPlacement placement = m_ptrCube->getPlacement();
	placement.setOrigin(projectPoint);

	//使洞口与cube宽度一致、延申方向一致
	m_ptrOpenning->setWidth(m_ptrCube->getWidth());
	m_ptrOpenning->setPlacement(placement);

	BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
	ptrGraphic = m_ptrOpenning->createPhysicalGraphics(*pProject, pModel->getModelId(), true);
	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphic, placement.toTransform());		

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enFlash);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphic->getEntityR());
}

bool ToolLayoutOpenningDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

//对工具进行注册
BPTool* CreateOpenningTool()
{
	ToolLayoutOpenningDemo* tool = new ToolLayoutOpenningDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutOpeningDemo", &CreateOpenningTool);
AutoDoRegisterFunctionsEnd