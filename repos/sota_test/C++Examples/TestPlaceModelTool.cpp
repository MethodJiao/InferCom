#include "stdafx.h"
#include "TestPlaceModelTool.h"
#include "DlgPlaceModel.h"

using namespace ::p3d::platform;

TestPlaceModelTool::TestPlaceModelTool()
{
}

TestPlaceModelTool::~TestPlaceModelTool()
{

}

bool TestPlaceModelTool::_onResetButton(::BIMBase::Core::BPBaseButtonEventCP ev)
{
	_exitTool();
	return true;
}

bool TestPlaceModelTool::_onDataButton(BPBaseButtonEventCP ev)
{
	//左键布置模型
	BPProjectP project = BPProject::getActiveProject();
	BPModelPtr modelPtr = BPModel::getActiveModel();
	if (!ev)
		return false;
	GePoint3d m_posStart = *ev->getPoint();
	GeTransform trans = GeTransform::createIdentityMatrix();
	trans.setTranslation(m_posStart);
	if (!__addModelToProject(trans, ev))
		return false;
	return true;
}

bool TestPlaceModelTool::_onInstall()
{
	return true;
}

void TestPlaceModelTool::_onPostInstall()
{
	T_Super::_onPostInstall();
	TOOLSETTING_OPEN;
	PBMessageCenter::Send(P3DApi::P3DApplication::JsonMessage("PBBim.Message.ToolPrompt", Utf8String(L"点击左键布置或单击右键结束").c_str()));
}

void TestPlaceModelTool::_exitTool()
{
	T_Super::_exitTool();
}

void TestPlaceModelTool::_onReinitialize()
{

}

void TestPlaceModelTool::_onRestartTool()
{
	TestPlaceModelTool* newTool = new TestPlaceModelTool();
	newTool->installTool();
}

void TestPlaceModelTool::_onReceive(Utf8CP messageType, JsonValueCR messageDataObj)
{
	if (0 == strcmp(PBBim_MESSAGE_CloseToolSetting, messageType))
	{
		_exitTool();
	}
	else if (0 == strcmp(PBBim_MESSAGE_GetUIValues, messageType))
	{
		wstring oldLocation = m_location;
		wstring locationWst;
		locationWst=string2wstring(P3DGlobalVariableManager::getManager().GetValue("PlaceModelSelectLocation", ""));
		m_location = locationWst.c_str();
		BPProjectP project = BPProject::getActiveProject();

		GimGeBase::PBDLComponentPtr dlComponentTypePtr;
		wstring wstModelPath = m_location.GetString();
		std::wstring wbfaName = wstModelPath.substr(wstModelPath.find_last_of(L"\\") + 1);
		std::string bfaName;
		bfaName = wstring2string(wbfaName);
		//从类型管理器中获得已经导入的类型
		std::map<std::string, std::vector<std::string>> allmodelType = GimGeBase::PEGlobalDLComponentPropManager::getTypeNames(project);
		auto bfaIter = allmodelType.find(bfaName);

		//如果类型管理器中不存在该类型，则导入
		if (bfaIter == allmodelType.end())
		{
			std::vector<Data::BPObjectPtr> importObjs;
			string strPath;
			strPath = wstring2string(wstModelPath);
			bool res = GimGeBase::PEGlobalDLComponentPropManager::importBfa(strPath, importObjs);
			if (res == false || importObjs.size() < 1)
				return;

			Data::BPObjectPtr object = importObjs.at(0);
			if (!object.IsValid())
				return;
			BPDataKey key = object->getDataKey();
			dlComponentTypePtr = dynamic_cast<GimGeBase::PBDLComponentP>(object.get());
			if (!dlComponentTypePtr.isValid())
				return;
		}
		else//如果类型管理器中存在该类型，直接从里面获取
		{
			dlComponentTypePtr = GimGeBase::PEGlobalDLComponentPropManager::getTypeByBfaAndTypeName(project, bfaIter->first, bfaIter->second[0]);
			if (!dlComponentTypePtr.isValid())
				return;
		}
		m_ptrGraphic = GimGeBase::PEGlobalDLComponentPropManager::getGraphicsFromNoumenon(dlComponentTypePtr, project);
		if (!m_ptrGraphic)
		{
			return;
		}
	}

}

bool TestPlaceModelTool::_onModelMotion(BPBaseButtonEventCP ev)
{
	__super::_onModelMotion(ev);
	//如果动态没有开启则开启动态，这样才可以进入_OnDynamicFrame函数
	if (!getDynamicsStarted())
		_beginDynamics();
	return true;
}

void TestPlaceModelTool::_onDynamicFrame(BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;
	BPProjectP project = BPProject::getActiveProject();
	GePoint3d evpoint = *ev->getPoint();
	GeTransform trans = GeTransform::createIdentityMatrix();
	trans.setTranslation(evpoint);

	BPRedrawEntitys redrawElems;
	redrawElems.SetDrawMode(BPDrawMode::enTempDraw);
	redrawElems.SetDrawPurpose(BPDrawPurpose::enDynamics);
	redrawElems.SetDynamicsViews(BPViewManager::getInstance(), BPViewManager::getInstance().getActivedViewport());
	redrawElems.setTransform(&trans);
	redrawElems.DoRedraw(m_ptrGraphic->getEntityR());
}

GimGeBase::PBDLComponentPtr TestPlaceModelTool::__addModelToProject(GeTransform trans, ::BIMBase::Core::BPBaseButtonEventCP ev)
{
	BPProjectP project = BPProject::getActiveProject();
	BPModelPtr modelPtr = BPModel::getActiveModel();

	wstring wstModelPath = m_location.GetString();

	std::wstring wbfaName = wstModelPath.substr(wstModelPath.find_last_of(L"\\") + 1);
	std::string bfaName;
	bfaName = wstring2string(wbfaName);
	std::map<std::string, std::vector<std::string>> allmodelType = GimGeBase::PEGlobalDLComponentPropManager::getTypeNames(project);
	auto bfaIter = allmodelType.find(bfaName);

	//类型管理器中不存在模型，重新导入
	if (bfaIter == allmodelType.end())
	{
		std::vector<Data::BPObjectPtr> importObjs;
		string strPath;
		strPath = wstring2string(wstModelPath);
		bool res = GimGeBase::PEGlobalDLComponentPropManager::importBfa(strPath, importObjs);
		if (res == false || importObjs.size() < 1)
			return false;

		Data::BPObjectPtr object = importObjs.at(0);
		if (!object.IsValid())
			return false;
		BPDataKey key = object->getDataKey();
		GimGeBase::PBDLComponentPtr dlComponentTypePtr = dynamic_cast<GimGeBase::PBDLComponentP>(object.get());
		if (!dlComponentTypePtr.isValid())
			return false;
		dlComponentTypePtr->setTransform(trans);
		//生成实例对象
		BPObjectPtr createObjectNew = BPParametricComponentTypeManager::createInstance(dlComponentTypePtr, trans, modelPtr->GetModelId(), project);
		BPGraphicElementPtr elePtr = dynamic_cast<BPGraphicElementP>(createObjectNew.get());
		if (!elePtr.isValid())
			return GimGeBase::PBDLComponentPtr();
		return dynamic_cast<GimGeBase::PBDLComponentP>(createObjectNew.get());
	}
	else
	{
		GimGeBase::PBDLComponentPtr dlComponentTypePtr = GimGeBase::PEGlobalDLComponentPropManager::getTypeByBfaAndTypeName(project, bfaIter->first, bfaIter->second[0]);
		if (!dlComponentTypePtr.isValid())
			return false;
		dlComponentTypePtr->setTransform(trans);
		//生成实例对象
		BPObjectPtr createObjectNew = BPParametricComponentTypeManager::createInstance(dlComponentTypePtr, trans, modelPtr->GetModelId(), project);
		BPGraphicElementPtr elePtr = dynamic_cast<BPGraphicElementP>(createObjectNew.get());
		if (!elePtr.isValid())
			return GimGeBase::PBDLComponentPtr();
		return dynamic_cast<GimGeBase::PBDLComponentP>(createObjectNew.get());
	}
}

//工具注册
BPTool* PlaceModelTool()
{
	TestPlaceModelTool* tool = new TestPlaceModelTool();
	return tool;
}

AutoDoRegisterFunctionsBegin //自动启动注册工具
BPToolsManager::registerTool("TestPlaceModelTool", &PlaceModelTool);
AutoDoRegisterFunctionsEnd //自动结束