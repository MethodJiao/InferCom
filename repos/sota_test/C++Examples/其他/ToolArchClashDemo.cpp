#include "stdafx.h"
#include "ToolArchClashDemo.h"
#include "EntitySymbologyEventDemo.h"

#pragma  comment(lib, "tgge.lib")
#pragma comment(lib, "BPClashDetection.lib")

using namespace ::p3d::platform;

ToolArchClashDemo::ToolArchClashDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集	
}


ToolArchClashDemo::~ToolArchClashDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->clear();
}

Utf8CP ToolArchClashDemo::_getToolName() const
{
	return "ArchClash";
}

void ToolArchClashDemo::_onPostInstall()
{
	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择图像,右键确定"));
	T_Super::_onPostInstall();
	BPSnap::getInstance().enableSnap(true);
}

void   ToolArchClashDemo::_onRestartTool()
{
	ToolArchClashDemo* newTool = new ToolArchClashDemo();
	newTool->installTool();
}

void ToolArchClashDemo::_exitTool()
{
	EntitySymbologyEventDemo::Get().end();
	__super::_exitTool();
}

bool ToolArchClashDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);
	EntitySymbologyEventDemo::Get().end();
	return true;
}

bool ToolArchClashDemo::_onResetButton(BPBaseButtonEventCP ev)
{
	// 获取当前工程
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (!pProject)
		return nullptr;

	BPViewportP pViewport = ev->getViewport();
	if (!pViewport)
		return false;

	if (getEntityArray()->getCount() == 0)
	{
		_exitTool();
		pViewport->forceHealImmediate();
		return true;
	}

	BPModelBaseP pModel = pProject->getActiveModel();
	if (!pModel)
		return false;

	m_vcEEH.clear();
	for (int i = 0; i < getEntityArray()->getCount(); i++)
	{
		m_vcEEH.push_back(getEntityArray()->getByIndex(i));
	}

	if (m_vcEEH.empty())
	{
		AfxMessageBox(L"未检测到碰撞！");
	}

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

	set<BPEntityId> clashId{};
	for (const auto& clashPair : clashRes)
	{
		clashId.insert(clashPair.first->getEntityId());
		clashId.insert(clashPair.second->getEntityId());
	}

	EntitySymbologyEventDemo::Get().setSelected(clashId);
	EntitySymbologyEventDemo::Get().begin();
	pViewport->forceHealImmediate();
	return true;
}

p3d::StatusInt ToolArchClashDemo::_onEntityModify(BPEntityR el)
{
	return ERROR;
}


BPTool* CreateToolArchClashDemo()
{
	ToolArchClashDemo* tool = new ToolArchClashDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("ArchClashDemo", &CreateToolArchClashDemo);
AutoDoRegisterFunctionsEnd