#include "stdafx.h"



using namespace DemoObject;


void DemoObject::ToolLayoutIndependentBridge::_onPostInstall()
{
	m_gPts.clear();
	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"点击确定构件原点位置/右键退出命令"));
}

void DemoObject::ToolLayoutIndependentBridge::_onRestartTool()
{
	m_gPts.clear();
	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"点击确定构件原点位置/右键退出命令"));
}

bool DemoObject::ToolLayoutIndependentBridge::_onDataButton(BIMBase::Core::BPBaseButtonEventCP ev)
{
	if (1 == m_gPts.size())
	{
		m_gPts.push_back(*ev->getPoint());
		BPModelPtr ptrModel = ev->getViewport()->getTargetModel();
		if (ptrModel.isNull())
		{
			m_gPts.clear();
			return false;
		}
		BPProjectP pProject = ptrModel->getBPProject();
		if (pProject == nullptr)
			return false;
		IndependentBridgePtr ptrIb = __createIB(m_gPts.at(0), m_gPts.at(1));
		if (ptrIb.isNull())
		{
			m_gPts.clear();
			return false;
		}
		if (0 != ptrIb->addToProject(*pProject, ptrModel->getModelId()))
		{
			m_gPts.clear();
			return false;
		}
		m_gPts.clear();
		PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"点击确定构件原点位置/右键退出命令"));
		return true;
	}

	m_gPts.push_back(*ev->getPoint());
	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"点击第二点确定独立桥架布置方向/右键重新布置"));
	return true;
}

bool DemoObject::ToolLayoutIndependentBridge::_onResetButton(BIMBase::Core::BPBaseButtonEventCP ev)
{
	if (0 == m_gPts.size())
	{
		_exitTool();
		return false;
	}
	m_gPts.clear();
	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"点击确定构件原点位置/右键退出命令"));
	return true;
}

void DemoObject::ToolLayoutIndependentBridge::_onDynamicFrame(BIMBase::Core::BPBaseButtonEventCP ev)
{
	if (1 != m_gPts.size())
		return;

	BPModelPtr ptrModel = ev->getViewport()->getTargetModel();
	if (ptrModel.isNull())
		return;

	BPProjectP pProject = ptrModel->getBPProject();
	if (pProject == nullptr)
		return;
	IndependentBridgePtr ptrIb = __createIB(m_gPts.at(0), *ev->getPoint());
	if (ptrIb.isNull())
		return;
	PModelId modleid = ptrModel->getModelId();
	
	
	GeTransform trans = ptrIb->getTransform();
	BPGraphicsPtr ptrGraphic = ptrIb->createPhysicalGraphics(*pProject, modleid, true);
	if (ptrGraphic.isNull())
		return;
	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphic, trans);
	ptrGraphic->finish();
	BPRedrawEntitys redrawElems;
	redrawElems.setDrawMode(BPDrawMode::enTempDraw);
	redrawElems.setDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.setDynamicsViews(ev->getViewport());
	redrawElems.doRedraw(ptrGraphic->getEntityR());
}

bool DemoObject::ToolLayoutIndependentBridge::_onModelMotion(BIMBase::Core::BPBaseButtonEventCP ev)
{
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

IndependentBridgePtr DemoObject::ToolLayoutIndependentBridge::__createIB(GePoint3d ptStart, GePoint3d ptEnd)
{
	IndependentBridgePtr ptrIb = new IndependentBridge();
	double dSpan = ptStart.distance(ptEnd);
	ptrIb->setCSSLong(dSpan);
	ptrIb->setBridgeArchHight(dSpan / 6);
	BPPlacement pl;
	pl.setPlacement(ptStart, GeVec3d::create(0, 0, 1), GeVec3d::create(ptEnd - ptStart));
	ptrIb->setPlacement(pl);
	return ptrIb;
}


BPTool* CreateToolLayoutIndependentBridge()
{
	ToolLayoutIndependentBridge* tool = new ToolLayoutIndependentBridge();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutIndependentBridgeDemo", &CreateToolLayoutIndependentBridge);
AutoDoRegisterFunctionsEnd
