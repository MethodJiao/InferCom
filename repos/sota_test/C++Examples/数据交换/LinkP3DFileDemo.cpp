#include "stdafx.h"
#include "LinkP3DFileDemo.h"
#include "BPDataCore/BPFileLinkElement.h"
#include "ExampleUtilDemo.h"
#include "BPDataCore/BPLinkFileManager.h"

//获取链接model
void getLinkFiles()
{
	vector<BIMBase::Data::NestingLinkFiles> vctLinkFile = BPLinkFileManager::getInstance().getNestingLinkFiles();
	if (vctLinkFile.size() > 0) 
	{
		std::map<::BIMBase::Core::BPProjectHandle, BPLinkFilePtr>           link = vctLinkFile[0].getLinkFiles();
		std::map<::BIMBase::Core::BPProjectHandle, BPLinkFilePtr>::iterator it = link.begin();
		for (; it != link.end(); it++)
		{
			BPProjectHandle projectHandle = it->second->getProjectHandle();

			BPProjectP pProject =
				BPApplication::getInstance().getProjectManager()->getProjectByHandle(projectHandle);
			if (pProject == NULL)
				continue;
			std::vector<BIMBase::Core::BPModelLinkPtr> modelLinks;
			BPLinkFileManager::getInstance().getAllModelLinkCollection(modelLinks, projectHandle);
			for (auto model : modelLinks)
			{
				BPModelPtr      mo = model->getOriginalModel();
				BPEntityVectorP entityVecP = mo->getGraphicEntitys();
				for (auto curr : *entityVecP)
				{
					if (!curr || !curr.isValid())
					{
						continue;
					}
					BPDataKey entikey = BPDataUtil::getDataKeyOnEntity(*curr);
					if (!entikey.isValid())
					{
						continue;
					}

					BPEntityId curentiid = curr->getEntityId();
				}
			}
		}
	}
		AfxMessageBox(_T("运行成功"));
}
//链接p3d做显示参照
void showRenceferce()
{
	IBPProjectManagerP promanager = BPApplication::getInstance().getProjectManager();
	if (promanager == NULL)
		return;
	//当前打开工程的project
	BPProjectP pRootProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pRootProject == nullptr)
		return ;

	wchar_t* str = ExampleUtilDemo::getProjectPathCW();

	PString appPathroot(str);
	PString filefullName = appPathroot + L"\\testfile\\cube.p3d";
	
	P3DStatus status;
	BPProjectP linkproject = promanager->openProject(status, filefullName, BPProject::BPOpenMode::enReadWrite);
	if (status == P3DStatus::ERROR || linkproject == NULL)
	{
		AfxMessageBox(_T("未找到链接工程"));
		return;
	}
		
	BPModelArray modelarray = linkproject->getAllModelsCollection();
	for (auto model : modelarray)
	{
		BPModelBasePtr modelref = model;
		PModelId modleid = modelref->getModelId();
		//只测试modelId等于0的
		if(modleid != 0)
			continue;

		BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
		if (NULL == pViewPort)
			return;
		BPModelPtr ptrRootModel = pViewPort->getRootModel();
		if (!ptrRootModel.isValid())
			return;
		BPModelLinkPtr link =  ptrRootModel->createModelLink(linkproject, modelref->getModelName());


	}
	AfxMessageBox(_T("运行成功"));
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("getLinkFilesDemo"), &getLinkFiles);
BPToolsManager::registerFun(_T("showRenceferce"), &showRenceferce);
AutoDoRegisterFunctionsEnd
