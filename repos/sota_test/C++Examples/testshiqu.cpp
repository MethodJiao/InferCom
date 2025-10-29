#include "stdafx.h"
#include "testshiqu.h"

USING_NAMESPACE_PBBIM_CORE
USING_NAMESPACE_PBBIM

TestGetProperty::TestGetProperty()
{
}

TestGetProperty::~TestGetProperty()
{

}

bool TestGetProperty::_onResetButton(::BIMBase::Core::BPBaseButtonEventCP ev)
{
	_exitTool();
	return true;
}
bool TestGetProperty::_onDataButton(::BIMBase::Core::BPBaseButtonEventCP ev)
{
	//使用该拾取工具前需要在工程中布置一个参数化组件实例对象，然后启动该拾取工具，左键拾取已经布置下去的模型，并修改这个对象的一条实例属性，并导出为bfa模型
	P_Super::_onDataButton(ev);
	BPProjectP project = BPProject::getActiveProject();
	BPEntityArrayPtr elementArray = _GetElemAgendaP();
	if (!elementArray.isValid())
		return false;
	for (int i = elementArray->size() - 1; i >= 0; i--)
	{
		BPEntityPtr entityPtr = elementArray->getByIndex(i);
		if (!entityPtr)
		{
			elementArray->erase(i);
			continue;
		}
		IBPObjectPtr ipbObjectPtr = BPObjectExtensionManager::Get().getBPObject(*entityPtr);
		if (!ipbObjectPtr)
		{
			elementArray->erase(i);
			continue;
		}
		GimGeBase::PBDLComponentPtr dlCom = dynamic_cast<GimGeBase::PBDLComponentP>(ipbObjectPtr.get());
		if (!dlCom.isValid())
			continue;
		//拿到属性分组,bool为true时说明该组属性是类型属性，bool为false为实例属性
		std::map<std::wstring, bool> groupNames = dlCom->getAllPropGroups();
		std::map <std::wstring, std::vector<std::wstring>> typeProps;//放类型属性的容器
		std::map <std::wstring, std::vector<std::wstring>> InsProps;//放实例属性的容器
		for (auto groupName : groupNames)
		{
			//根据属性分组拿到属性的key
			std::vector<std::wstring>propKeys = dlCom->getAllPropByGroupName(groupName.first);
			if (groupName.second == true)
				typeProps.insert(make_pair(groupName.first, propKeys));//类型属性的groupname和属性key放进去
			else
				InsProps.insert(make_pair(groupName.first, propKeys));//实例属性的groupname和属性key放进去
			for (auto propKey : propKeys)
			{
				//根据属性的分组和属性的key拿到对应的这一条属性条目
				GimGeBase::PEPropertyItem item = dlCom->getPropertyByKey(propKey, groupName.first);
				std::wstring originValue = L"";
				//拿到这条属性条目中的属性值并转化为字符串
				item.getStringValue(originValue);
				//修改实例属性的值，以工程属性中的“数量”为例
				if (propKey == L"数量")
				{
					std::wstring value = L"5个";
					//重新设置这条属性的属性值
					item.setStringValue(value);
					//修改这条实例属性的值
					dlCom->editPropertyByKey(propKey, item, groupName.first);
					//将数据持久化
					dlCom->update(project);
				}
			}
		}
		//根据该实例对象获得他的类型对象
		GimGeBase::PBDLComponentPtr dlComType = dlCom->getDLComponentType();
		if (!dlComType.isValid())
			continue;
		std::map<std::wstring, std::vector<std::string>> partBfaNames;
		//获得这个类型对象所有的一级子对象的bfa名称和bfakey
		dlComType->getPartBfaNames(partBfaNames);
		std::wstring path = L"PLATFORM\\BPGroup\\ComponentLib\\变电\\电气\\变压器\\三绕组变压器\\110kV变压器模型新.bfa";
		std::wstring bfaPath = BIMBase::Core::BPApplication::getInstance().getAppPath().c_str() + path;
		std::string strBfaPath = wstring2string(bfaPath);
		//将bfa模型导出到指定路径下
		if (!GimGeBase::PEGlobalDLComponentPropManager::exportBfa({ dlComType }, strBfaPath))
			return false;
	}
	return true;
}
bool TestGetProperty::_onInstall()
{
	return true;
}
void TestGetProperty::_onPostInstall()
{
	P_Super::_onPostInstall();
	TOOLSETTING_OPEN;
}
void TestGetProperty::_exitTool()
{
	P_Super::_exitTool();
}
void TestGetProperty::_onReinitialize()
{

}
void TestGetProperty::_onRestartTool()
{
	TestGetProperty* newTool = new TestGetProperty();
	newTool->installTool();
}

void  TestGetProperty::_onReceive(Utf8CP messageType, JsonValueCR messageDataObj)
{

}

//工具注册
BPTool* PlaceTestGetProperty()
{
	TestGetProperty* tool = new TestGetProperty();
	return tool;
}

AutoDoRegisterFunctionsBegin //自动启动注册工具
BPToolsManager::registerTool("TestGetProperty", &PlaceTestGetProperty);
AutoDoRegisterFunctionsEnd //自动结束