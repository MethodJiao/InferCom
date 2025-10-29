#include "stdafx.h"
#include "ToolClashDemo.h"
#include "EntitySymbologyEventDemo.h"

#pragma  comment(lib, "tgge.lib")
#pragma comment(lib, "BPClashDetection.lib")

using namespace ::p3d::platform;

ToolClashDemo::ToolClashDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集	
}


ToolClashDemo::~ToolClashDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->clear();
}

void ToolClashDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPSnap::getInstance().enableSnap(true);
}

void   ToolClashDemo::_onRestartTool()
{
	ToolClashDemo* newTool = new ToolClashDemo();
	newTool->installTool();
}

void ToolClashDemo::_exitTool()
{	
	EntitySymbologyEventDemo::Get().end();
	__super::_exitTool();
}

bool ToolClashDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);
	EntitySymbologyEventDemo::Get().end();
	return true;
}

bool ToolClashDemo::_onResetButton(BPBaseButtonEventCP ev)
{
	
	// 获取当前工程
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;

	BPViewportP pViewport = ev->getViewport();
	if (pViewport == nullptr)
		return false;

	if (getEntityArray()->getCount() == 0)
	{
		_exitTool();
		pViewport->forceHealImmediate();
		return true;
	}

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return false;


	m_vcEEH.clear();
	for (int i = 0; i < getEntityArray()->getCount(); i++)
	{
		m_vcEEH.push_back(getEntityArray()->getByIndex(i));
	}

    //BPModelBaseP pModel = pProject->getActiveModel();

	ClashRule rule;
	rule.m_vecEntity = m_vcEEH;

	getEntityArray()->clear();

	//获取model的transform
	BPModelLinkArrayCR vctLink = pModel->getAllModelLinks();
	for_each(vctLink.begin(), vctLink.end(), [&](BPModelLinkPtr modelLink)
	{
		GePoint3d point;
		modelLink->getOrigin(point);
		GeRotMatrix rotMatrix = modelLink->getRotMatrix();
		double dScale = modelLink->getScale();
		GeTransform trans = GeTransform::create(rotMatrix, point);
		trans.setByScaleMatrixColumns(dScale, dScale, dScale);
		rule.m_mapModelTransform.insert(make_pair(modelLink->getModelId(), trans));
	});

	PBBim::PBCD::CDFunction::SetCDSafeDis(1000);//设置碰撞距离
	ClashMethod clashMethod;
	clashMethod.doClash(rule);
	ClashResult clashRes;
	clashMethod.getClashResult(clashRes);

	set<BPEntityId>  clashId;
	for (auto clashPair : clashRes)
	{
		clashId.insert(clashPair.first->getEntityId());
		clashId.insert(clashPair.second->getEntityId());
	}

	EntitySymbologyEventDemo::Get().setSelected(clashId);
	EntitySymbologyEventDemo::Get().begin();
	pViewport->forceHealImmediate();
	return true;
}

p3d::StatusInt ToolClashDemo::_onEntityModify(BPEntityR el)
{
	return ERROR;
}


BPTool* CreateToolClashDemo()
{
	ToolClashDemo* tool = new ToolClashDemo();
	return tool;
	return NULL;
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("clashDemo", &CreateToolClashDemo);
AutoDoRegisterFunctionsEnd