#include "stdafx.h"
#include <ctime>
#include "ToolExportDwgDemo.h"
#include "BPUIFrameWork/BPProgressBall.h"
#include "OdWriteEx/OdWriteEx.h"
#include "direct.h"
#include "ExampleUtilDemo.h"

#pragma comment(lib, "BPWriteReadEx.lib")
static BIMBase::FrameWork::BPProgressBall* s_proxy = nullptr;
BOOL readDwgFile(LPCTSTR fileName,
	PBModelInfoR model,
	GeTransformCR matrix = GeTransform::createIdentityMatrix(),
	COLORREF     backColor = RGB(0, 0, 0),
	void(*ptrMeterProgressFun)(int Pos) = nullptr,
	bool bIsMTMode = false,
	bool bEnablePartialLoading = false,
	bool bDisableSvcsOutput = false,
	bool bDsableRecompute = false,
	bool bDsableDump = false,
	bool bEnableAcisAudit = false
);


Utf8CP ToolExportDwgDemo::getToolName()
{ 
	return "ExportDwg"; 
}

void ToolExportDwgDemo::doImportDwgDemo()
{
	BIMBase::FrameWork::ProgressCtrlTypeInfo info;
	info.b_DisplayPercent = false;
	BIMBase::FrameWork::BPProgressBall proxy(info);
	std::wstring sText = _T("导入dwg文件中");
	proxy.setPBProgressStatusText(sText);
	
	void(*ptrMeterProgressFun)(int nPos) = [](int nPos)
	{
		BIMBase::FrameWork::ProgressCtrlTypeInfo info;
		info.b_DisplayPercent = false;
		if (s_proxy == nullptr)
		{
			std::wstring sText = _T("导入dwg文件中");
			s_proxy = new BIMBase::FrameWork::BPProgressBall(info);
			s_proxy->setPBProgressStatusText(sText);
		}
		if (nPos < 100 && s_proxy)
		{
			s_proxy->setPBProgressPos(nPos);
		}
		else if (nPos == 101)
		{
			s_proxy->close();
			delete s_proxy;
			s_proxy = nullptr;
		}
		
	};

	BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
	if (NULL == pViewPort)
	{
		return;
	}

	BPModelP pModel = pViewPort->getTargetModel();
	PModelId modelId = pModel->getModelId();

	PBModelInfoPtr ptrModelInfo = PBModelInfoManager::Get().GetModelById(modelId);
	if (ptrModelInfo == nullptr)
		return;

	GeTransform tran;
	tran.setByIdentityMatrix();
	GePoint3d pt = GePoint3d::create(0, 0, 0);
	tran.setByOriginAndVectors(pt, GeVec3d::create(1, 0, 0), GeVec3d::create(0, 1, 0), GeVec3d::create(0, 0, 1));
	wchar_t* str = ExampleUtilDemo::getProjectPathCW();

	P3DFileName sFilePath(str);
	sFilePath.appendToDir(L"testfile\\PKPM.dwg");

	BOOL bTemp = readDwgFile(sFilePath.c_str(), *ptrModelInfo, tran, RGB(255, 255, 25), ptrMeterProgressFun);
}

void ToolExportDwgDemo::doExportDwgDemo()
{

	//保存路径
	std::wstring wstrPath = _T("D:\\");

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
	
	BPModelP pModel = pViewPort->getTargetModel();
	PModelId modelId = pModel->getModelId();

	::p3d::pset<BPEntityId> elementIdSet;
	pvector<ppair<GeTransform, pset<BPEntityId>> > pairTransElementSet;

	if (P3DStatus::ERROR== BPEntityUtil::getEntitiesOfModel(elementIdSet, *pProject, modelId))
	{
		return;
	}

	if (!elementIdSet.empty())
	{
		PBBim::PBBimCore::PBBimDomain domain = PBUserEnvironment::getInstance()->getDefaultDomain();
		GeTransform trans = GeTransform::createIdentityMatrix();
		wstring wStrprojectName = wstrPath + getFileName();

		pairTransElementSet.push_back(make_pair(trans, elementIdSet));
		outPutDwgFilePure(wStrprojectName.c_str(), pairTransElementSet);
		AfxMessageBox(L"输出路径为D盘");
	}
}

std::wstring ToolExportDwgDemo::getFileName()
{
	tm t;   //tm结构指针
	time_t now;  //声明time_t类型变量
	time(&now);      //获取系统日期和时间
	localtime_s(&t, &now);   //获取当地日期和时间
	wstring str = L"outPutDwg";
	str = str + L"_" + to_wstring(t.tm_year + 1900) + L"_";	//年
	str = str + to_wstring(t.tm_mon + 1) + L"_";			//月
	str = str + to_wstring(t.tm_mday) + L"_";				//日
	str = str + to_wstring(t.tm_hour) + L"_";				//时
	str = str + to_wstring(t.tm_min) + L"_";				//分
	str = str + to_wstring(t.tm_sec);						//秒
	str += L".dwg";
	return str;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("exportDwgDemo"), &ToolExportDwgDemo::doExportDwgDemo);
BPToolsManager::registerFun(_T("importDwgDemo"), &ToolExportDwgDemo::doImportDwgDemo);
AutoDoRegisterFunctionsEnd