#include "stdafx.h"
#include "BPUIFrameWork\QSMessageExchange.h"

void exportIFCFun()
{
	HMODULE hmod = ::LoadLibrary(_T("IfcDataExTool.dll"));
	if (hmod)
	{
		/*bool bS = DataExchange::IfcExportUtil::exportAll2Ifc();
		::FreeLibrary(hmod);*/
		AfxMessageBox(L"保存成功");
		BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
		if (pProject == NULL)
		{
			return;
		}
		BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
		if (NULL == pViewPort)
		{
			return;
		}
		PString wPath = L"D:\\ifc\\aaa.ifc";
		BPModelP pModel = pViewPort->getTargetModel();
		PModelId modelId = pModel->getModelId();
		::p3d::pset<BPEntityId> elementIdSet;
		pvector<ppair<GeTransform, pset<BPEntityId>> > pairTransElementSet;

		if (P3DStatus::ERROR == BPEntityUtil::getEntitiesOfModel(elementIdSet, *pProject, modelId))
		{
			return;
		}
		pvector<::BIMBase::PModelId> modelIds;
		modelIds.push_back(modelId);
		bool ST = DataExchange::IfcExchaneUtil::exportAll2Ifc(modelIds, wPath);
	}


	
}


void importIFCFun()
{
	HMODULE hmod = ::LoadLibrary(_T("IfcDataExTool.dll"));
	if (hmod)
	{
		bool bS = DataExchange::IfcExportUtil::importAll2Ifc();
		::FreeLibrary(hmod);
	}
}

//静默启动的方式导出ifc
class CSQSMessage :public BIMBase::FrameWork::QSMessageBase
{
public:
	CSQSMessage()
	{
		AfxMessageBox(L"保存成功hello");
	}
protected:
	virtual  std::wstring _runCommond(std::wstring data)
	{
		HMODULE hmod = ::LoadLibrary(_T("IfcDataExTool.dll"));
		if (hmod)
		{
			BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
			if (pProject == NULL)
			{
				return L"";
			}
			BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
			if (NULL == pViewPort)
			{
				return L"";
			}
			PString wPath = L"D:\\ifc\\aaa.ifc";
			BPModelP pModel = pViewPort->getTargetModel();
			PModelId modelId = pModel->getModelId();
			::p3d::pset<BPEntityId> elementIdSet;
			pvector<ppair<GeTransform, pset<BPEntityId>> > pairTransElementSet;

			if (P3DStatus::ERROR == BPEntityUtil::getEntitiesOfModel(elementIdSet, *pProject, modelId))
			{
				return L"";
			}
			pvector<::BIMBase::PModelId> modelIds;
			modelIds.push_back(modelId);
			bool ST = DataExchange::IfcExchaneUtil::exportAll2Ifc(modelIds, wPath);
		}
		return L"";
	}

};
CSQSMessage s_test;
AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("exportIFCDemo", exportIFCFun);
BPToolsManager::registerFun("importIFCDemo", importIFCFun);
BIMBase::FrameWork::MessageRegisterUtil::registerCommond(L"ExportIFC", &s_test);
AutoDoRegisterFunctionsEnd
