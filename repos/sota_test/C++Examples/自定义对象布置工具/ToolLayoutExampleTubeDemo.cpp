#include "stdafx.h"


using namespace DemoObject;

ToolLayoutExampleTubeDemo::ToolLayoutExampleTubeDemo()
{
	m_gPts.clear();
}

ToolLayoutExampleTubeDemo::~ToolLayoutExampleTubeDemo()
{
	m_gPts.clear();
}

void ToolLayoutExampleTubeDemo::_onPostInstall()
{
	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择起点"));
	m_gPts.clear();
}

void ToolLayoutExampleTubeDemo::_onRestartTool()
{

	m_gPts.clear();
}

bool ToolLayoutExampleTubeDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;

	BPModelBaseP pModelref = pProject->getActiveModel();
	if (pModelref == nullptr)
		return false;
	
	switch (m_gPts.size())
	{
	case 0:
	{
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择终点"));
		m_gPts.push_back(*ev->getPoint());
		break;
	}
	
	case 1:
	{
		BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择起点"));
		m_gPts.push_back(*ev->getPoint());

		ExampleTubeDemoPtr ptrTube = new ExampleTubeDemo(m_gPts.at(0), m_gPts.at(1), 100, 5);
		if (ptrTube == nullptr)
			return false;

		BPPlacement placement;
		placement.setPlacement(m_gPts.at(0), m_gPts.at(1), 1, 0);
		ptrTube->setPlacement(placement);

		if (0 != ptrTube->addToProject(*pProject, pModelref->getModelId()))
			return false;

		m_gPts.clear();
		break;
	}
	default:
		break;
	}

	return true;
}

bool ToolLayoutExampleTubeDemo::_onResetButton(BPBaseButtonEventCP ev)
{
	if (0 == m_gPts.size())
	{
		_exitTool();
		return false;
	}
	m_gPts.clear();
	return true;
}

void ToolLayoutExampleTubeDemo::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (m_gPts.size() == 0)
		return;

	GePoint3d ptDynamic = *ev->getPoint();

	if (m_gPts.at(0).distance(ptDynamic) < 10)
		return;

	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == nullptr)
		return;
	PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();

	ExampleTubeDemoPtr ptrTube = new ExampleTubeDemo(m_gPts.at(0), ptDynamic, 100, 5);
	if (ptrTube == nullptr)
		return;

	BPPlacement placement;
	placement.setPlacement(m_gPts.at(0), ptDynamic, 1, 0);
	ptrTube->setPlacement(placement);
	GeTransform trans = ptrTube->getTransform();
	BPGraphicsPtr ptrGraphic = ptrTube->createPhysicalGraphics(*pProject, curModelId, true);
	if (ptrGraphic.isNull())
		return;

	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());

	//把工程图的图素单独拿出来transform
	BPEntityArray array = ptrGraphic->getElementArray();
	for (int i = 0;i < array.getCount();i++)
	{
		BPEntityPtr ptrEntity = array.getByIndex(i);
		if(ptrEntity.isNull())
			continue;
		ptrEntity->setTransfrom(trans);
		redrawElems.doRedraw(*ptrEntity);
	}

	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphic, trans);

	if (ptrGraphic == nullptr)
		return;
	ptrGraphic->finish();
	
	redrawElems.doRedraw(ptrGraphic->getEntityR());
	
}

bool ToolLayoutExampleTubeDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}



BPTool* CreateLayoutExampleTube()
{
	ToolLayoutExampleTubeDemo* tool = new ToolLayoutExampleTubeDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutExaTubeDemo", &CreateLayoutExampleTube);
AutoDoRegisterFunctionsEnd