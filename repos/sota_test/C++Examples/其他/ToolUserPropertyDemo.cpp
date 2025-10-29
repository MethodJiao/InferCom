#include "stdafx.h"
#include "ToolUserPropertyDemo.h"
#include "share/Common/ExportMacro.h"
#include "share/Common/WDPublicDef.h"
#include "WDUi/BCAttributeManagement.h"
#include "WDUi/BCAttributeMgn.h"
#include "WDUi/BCAttributeUtil.h"


using namespace ::p3d::platform;

ToolUserPropertyDemo::ToolUserPropertyDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集	
}


ToolUserPropertyDemo::~ToolUserPropertyDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->clear();
}

void ToolUserPropertyDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPSnap::getInstance().enableSnap(true);
	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"点击对象"));
}

void   ToolUserPropertyDemo::_onRestartTool()
{
	ToolUserPropertyDemo* newTool = new ToolUserPropertyDemo();
	newTool->installTool();
}

void ToolUserPropertyDemo::_exitTool()
{

	__super::_exitTool();
}

BPEntityPtr ToolUserPropertyDemo::_buildLocateAgenda(BPPickDataCP pickData, BPBaseButtonEventCP ev)
{
	m_curEEH = T_Super::_buildLocateAgenda(pickData, ev);
	return m_curEEH;
}

bool ToolUserPropertyDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;

	BPViewportP pViewport = ev->getViewport();
	if (pViewport == nullptr)
		return false;

	if (m_curEEH.isNull())
	{
		AfxMessageBox(L"请选择对象");
		return false;
	}
		

	BPDataKey dataKey = BPDataUtil::getDataKeyOnEntity(*m_curEEH);
	IBPObjectPtr ptrIBPObj = BPObjectExtensionManager::getInstance().getBPObject(*pProject, dataKey);
	if (ptrIBPObj.isNull())
		return false;
	
	m_curObj = dynamic_cast<BPGraphicElementP>(ptrIBPObj.get());
	if (m_curObj.isNull())
		return false;

	Json::Value jsonValue;
	wd::BCAttributeUtil::getAttributeJson(jsonValue, m_curObj);
	Json::FastWriter jfw;
	CString sRet = (LPCTSTR)(_bstr_t)jfw.write(jsonValue).c_str();
	AfxMessageBox(sRet);
	return true;
}

static int s_index = 1;
bool ToolUserPropertyDemo::_onResetButton(BPBaseButtonEventCP ev)
{
	// 获取当前工程
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;
	//获取对象的自定义属性类型
	if (m_curObj.isNull())
	{
		_exitTool();
		return false;
	}
		
	std::wstring  sCategory = m_curObj->getClassNameCustom().c_str();
	BPExtendPropertySet eExtendPropertySet = m_curObj->getExtendPropertySet();
	BPPropertyValue valueOld;
	if (eExtendPropertySet.getSubParam(L"Category", valueOld))
	{
		if (valueOld.m_type == BPPropertyValue::String)
			sCategory = valueOld.m_value.m_valueString.c_str();
	}
	//在该类型自定义属性添加项
	std::map<PString, std::vector<wd::BCAttributeItemInfo>> mapItems;
	PString strCategory(sCategory.data());
	Utf8String u8;
	P3DStringHelper::wCharToUtf8(u8, strCategory.data());
	wd::BCAttributeManagement::instance().getAttributeItems(u8, mapItems);
	CString name = BPDomainEnvironment::getInstance()->getCurrentDomainKeyName();
	switch (s_index)
	{
	case 3:
	{
		wd::BCAttributeItemInfo addItem;
		addItem.sName = L"DemoAdd";
		addItem.val = BPValue(L"addItem");
		addItem.vctEnum.clear();
		bool sta = wd::BCAttributeManagement::instance().addAttributeItems(name.GetString(), u8, addItem);
		s_index++;
		break;
	}
	case 2:
	{
		wd::BCAttributeItemInfo addItem;
		addItem.sName = L"DemoAdd2";
		addItem.val = BPValue(2);
		addItem.vctEnum.clear();

		bool sta = wd::BCAttributeManagement::instance().addAttributeItems(name.GetString(), u8, addItem);
		s_index++;
		break;
	}
	case 1:
	{
		wd::BCAttributeItemInfo addItem;
		addItem.sName = L"DemoAdd3";
		addItem.val = BPValue(L"3.5");
		addItem.vctEnum = { L"2.6", L"3.5", L"4" };
		bool sta = wd::BCAttributeManagement::instance().addAttributeItems(name.GetString(), u8, addItem);
		s_index++;
		break;
	}
	case 4:
	{
		wd::BCAttributeItemInfo addItem;
		addItem.sName = L"DemoAdd4";
		addItem.val = BPValue(true);
		bool sta = wd::BCAttributeManagement::instance().addAttributeItems(name.GetString(), u8, addItem);
		s_index++;
		break;
	}
	default:
		break;
	}
	wd::BCAttributeUtil::RefreshUserPropertyElement();

	_exitTool();
	return true;
}

p3d::StatusInt ToolUserPropertyDemo::_onEntityModify(BPEntityR el)
{
	return ERROR;
}


BPTool* CreateToolUserPropertyDemo()
{
	ToolUserPropertyDemo* tool = new ToolUserPropertyDemo();
	return tool;
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("userPropertyDemo", &CreateToolUserPropertyDemo);
AutoDoRegisterFunctionsEnd

