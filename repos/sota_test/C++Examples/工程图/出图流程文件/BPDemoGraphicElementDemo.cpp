#include "stdafx.h"
#include "BPDrawingCuttingDemo.h"
#include "BPDemoGraphicElementDemo.h"



using namespace DemoObject;

BPDemoGraphicElementDemo::BPDemoGraphicElementDemo()
{
	
}

BPDemoGraphicElementDemo::~BPDemoGraphicElementDemo()
{

}

::p3d::P3DStatus BPDemoGraphicElementDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (p3d::P3DStatus::SUCCESS != BPGraphicElement::_copyToData(instance, project))
		return P3DStatus::ERROR;
	return P3DStatus::SUCCESS;
}

::p3d::P3DStatus BPDemoGraphicElementDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (p3d::P3DStatus::SUCCESS != BPGraphicElement::_initFromData(instance))
		return P3DStatus::ERROR;
	return P3DStatus::SUCCESS;
}

BIMBase::Core::BPGraphicsPtr BPDemoGraphicElementDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
{
	return nullptr;
}

BIMBase::Core::BPGraphicsPtr BPDemoGraphicElementDemo::_createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId)
{
	return nullptr;
}
BIMBase::Core::BPGraphicsPtr BPDemoGraphicElementDemo::createPhysicalGraphicsForDrawing(BIMBase::Core::BPProject& project, BIMBase::PModelIdCR modelId)
{
	return _createPhysicalGraphicsForDrawing(project, modelId);
}

::p3d::P3DStatus  BPDemoGraphicElementDemo::_addToProject(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId)
{
	//先判断活动视口有没有图纸model，如果没有就不需要剖切，直接调用addtoproject
	vector<int> activeViewSet;
	//如果有sheetmodel，那就要更新对应的剖切，调用剖切的函数
	vector<PModelId> drawModelId;//所生成的图纸model
	set<PModelId> setViewModelId;//所有可见
	BPEntityId idphysicalelement;
	BPViewManager::getInstance().getAllActiveViewports(activeViewSet);
	int count = 0;
	bool flag = false;
	for (int view : activeViewSet)
	{
		BPViewportP     pview = BPViewManager::getInstance().getViewport(view);
		if (pview == nullptr)
			continue;
		BPModelP model = pview->getTargetModel();
		if (model == nullptr)
			continue;
		PModelId moid = model->getModelId();
		setViewModelId.insert(moid);
		PBModelInfoPtr modelInfo = PBModelInfoManager::Get().GetModelById(moid);
		if (!modelInfo.isValid())
			continue;
		PBModelType type = modelInfo->GetModelType();
		if (type == PBModelType::Axis)//说明这个entity是三维上的
		{

		}
		else if (type == PBModelType::Sheet)//说明图纸model现在在视口中显示
		{
			//如果此时显示的窗口上是带有图框，那么在三维加东西，也不去更新图纸了
			if (modelInfo->GetDisplayedName() == L"Displaymodel")
				flag = true;

			if (modelId == moid)//说明此时是在2维上加
			{
				BPModelBaseP pModle = BPViewManager::getInstance().getViewport(0)->getTargetModel();
				if (pModle == nullptr)
					return P3DStatus::ERROR;;
				PModelId modelidd = pModle->getModelId();
				//在原始视口0上布置3维的
				if (SUCCESS != BPGraphicElement::_addToProject(project, modelidd))//加在3维上
				{
					AfxMessageBox(L"Can not add to project!");
				}
				 idphysicalelement = BPEntityUtil::getPrimaryEntityWithData(project, getDataKey(), modelidd);
			}
			else
			{
				if (SUCCESS != BPGraphicElement::_addToProject(project, modelId))//加在3维上
				{
					AfxMessageBox(L"Can not add to project!");
				}
				 idphysicalelement = BPEntityUtil::getPrimaryEntityWithData(project, getDataKey(), modelId);
				 flag = false;
			}
			count++;
		}

	}
	if (count == 0 || flag )//说明现在视口没有显示图纸，h或者显示的是带图框的图纸，也不用做下面的剖切了
	{
		::p3d::P3DStatus sta = BPGraphicElement::_addToProject(project, modelId);
		return sta;
	}

	if (!idphysicalelement.isValid())
		return P3DStatus::ERROR;

	T_PBModelInfoPtrArrayCR modelInfoArray = PBModelInfoManager::Get().GetAllModels();
	for (PBModelInfoPtr modelInfoPtr : modelInfoArray)
	{
		if (modelInfoPtr == nullptr)
			continue;
		PBModelType type = modelInfoPtr->GetModelType();
		if (type == PBModelType::Sheet)//说明是剖切生成得图纸
		{
			PModelId id = modelInfoPtr->GetModelId();
			drawModelId.push_back(id);
		}
	}

	for (auto modelid : drawModelId)
	{
		if (setViewModelId.find(modelid) == setViewModelId.end())
			continue;

		PBModelInfoPtr modelInfo = PBModelInfoManager::Get().GetModelById(modelid);//拿到二维图纸model
		if (!modelInfo.isValid())
			continue;
	
		//返回model去做剖切，通过拿到model上的三维东西
		BPModelP phiysicalmodel = project.loadModelById(idphysicalelement.m_modelId);
		BPEntity entity(idphysicalelement, project);
		if (!entity.isValid())
			continue;
		if (phiysicalmodel == nullptr )
			continue;
		BPDrawingParasManagerDemo::eDrawingview type = BPDrawingParasManagerDemo::Get().getDrawingview();
		BPDrawingCuttingDemo::Get().addElementToCut(&project, phiysicalmodel, entity, modelInfo, type);
		

	}
	return P3DStatus::SUCCESS;
}


::p3d::P3DStatus BPDemoGraphicElementDemo::_replaceInProject(::BIMBase::Core::BPProjectR project, bool bReCreateGeometry )
{
	::p3d::P3DStatus sta = BPGraphicElement::_replaceInProject(project);

	set<BPEntityId> elemId;
	::p3d::P3DStatus status = BPDataUtil::getAllBindingEntityFromData(elemId, getDataKey(), &project);
	if(status != P3DStatus::SUCCESS)
		return P3DStatus::ERROR;
	set<PModelId> setViewModelId;//所有可见
	vector<int> activeViewSet;
	BPViewManager::getInstance().getAllActiveViewports(activeViewSet);
	bool flag = false;
	for (int view : activeViewSet)
	{
		BPViewportP     pview = BPViewManager::getInstance().getViewport(view);
		if (pview == nullptr)
			continue;
		BPModelP model = pview->getTargetModel();
		if (model == nullptr)
			continue;
		PModelId moid = model->getModelId();
		setViewModelId.insert(moid);

		PBModelInfoPtr modelInfo = PBModelInfoManager::Get().GetModelById(moid);
		if (!modelInfo.isValid())
			continue;
		PBModelType type = modelInfo->GetModelType();
		if (type == PBModelType::Sheet)//说明图纸model现在在视口中显示
		{
			//如果此时显示的窗口上是带有图框，那么在三维加东西，也不去更新图纸了
			if (modelInfo->GetDisplayedName() == L"Displaymodel")
				flag = true;
		}
	}

	vector<BPEntityId> eleIdphy;
	vector<PModelId> drawModelId;//所生成的图纸model
	vector < PBSheetModelInfoP> sheetmodels;
	for (auto entityid : elemId)
	{
		PBModelInfoPtr modelInfo = PBModelInfoManager::Get().GetModelById(entityid.m_modelId);
		if (!modelInfo.isValid())
			continue;
		PBModelType type = modelInfo->GetModelType();
		if (type == PBModelType::Axis)//说明这个entity是三维上的
		{
			
			eleIdphy.push_back(entityid);
		}
		else if (type == PBModelType::Sheet)
		{
			PBSheetModelInfoP sheetmodel = dynamic_cast<PBSheetModelInfoP>(modelInfo.get());
			if (sheetmodel != nullptr)
			{
				PModelId modeid = sheetmodel->GetModelId();
				if (setViewModelId.find(modeid) != setViewModelId.end())
				{
					sheetmodels.push_back(sheetmodel);
					drawModelId.push_back(modeid);
				}
			}
			
		}
	}

	if (drawModelId.size() == 0 || flag)
		return sta;
	

	for (auto entityid : eleIdphy)
	{
		BPModelP phiysicalmodel = project.loadModelById(entityid.m_modelId);
		if (phiysicalmodel == nullptr)
			continue;
		//返回model去做剖切，通过拿到model上的三维东西
		
		BPEntity entity(entityid, project);
		if (!entity.isValid())
			continue;
		for (auto sheet : sheetmodels)
		{
			PString sName;
			PBModelInfoPtr modelInfo = sheet;
			if (!modelInfo.isValid())
				continue;
			sName = sheet->GetDisplayedName();
			
			BPDrawingParasManagerDemo::eDrawingview type = BPDrawingParasManagerDemo::Get().getDrawingview();
			BPDrawingCuttingDemo::Get().addElementToCut(&project, phiysicalmodel, entity, modelInfo, type);
		}

	}
	return sta;
}