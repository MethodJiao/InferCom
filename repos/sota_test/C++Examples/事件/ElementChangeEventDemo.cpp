#include "stdafx.h"
#include "ElementChangeEventDemo.h"
#include "CubeDemo.h"
#include "createEntitysLinkDemo.h"
using namespace PBBim::PBBimCore;
extern int g_newview ;
extern PModelId g_ModelId ;


ElementChangeEventDemo::ElementChangeEventDemo()
{
}


ElementChangeEventDemo::~ElementChangeEventDemo()
{
}

bool ElementChangeEventDemo::_onPostNew(BPEntityChangeEventArgCR arg)
{
	vector<int> vct;
	BPViewManager::getInstance().getAllActiveViewports(vct);
	bool bIsmutiviews = vct.size() == 1 ? false : true;
	//如果不是多视口，就直接返回
	if (!bIsmutiviews)
		return false;
	
	BPDataKey datakey = arg.getDataKey();
	PModelId modid = arg.getModelId();
	BPEntityId entid = arg.getEntityId();
	entid.m_modelId = modid;
	BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return false;

	BPEntity entity(entid, *pProject);
	if (!entity.isValid())
		return false;
	::p3d::PString className = L"";
	//多视口联动目前针对cube，其他对象就先不弄
	entity.getClassName(className);
	if (className != L"CubeDemo")
		return false;
	
	BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(entity);
	if (!ptrData.isValid())
		return false;

	DemoObject::CubeDemo pbCube;
	pbCube.initFromData(*ptrData);
	GeTransform trans = pbCube.getTransform();
	BPGraphicsPtr ptrGraps = pbCube.createGraphicsPlane(*pProject, g_ModelId);
	if (ptrGraps == nullptr)
		return false;
	//新建视口上只允许画2维的，如果是把3维的布置到新视口，那把这个删掉，然后在视口0上画3维
	if (modid == g_ModelId)//说明现在画的3维的是布置在视口1上
	{
		//把这个视口上的三维的cube删掉
		entity.deleteFromModel();
		BPModelBaseP pModle = BPViewManager::getInstance().getViewport(0)->getTargetModel();
		if (pModle == nullptr)
			return false;
		PModelId modelidd = pModle->getModelId();
		//在原始视口0上布置3维的
		if (SUCCESS != pbCube.addToProject(*pProject, modelidd))
		{
			AfxMessageBox(L"Can not add to project!");
		}
		datakey = pbCube.getDataKey();
	}
	
	
	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraps,trans);
	
	BPEntityId entityId = ptrGraps->save();
	BPEntity graentity(entityId, *pProject);
	if (!graentity.isValid())
		return false;

	P3DStatus sta = BPDataUtil::bindEntityToData(entityId, datakey, pProject);
	if (sta != P3DStatus::SUCCESS)
		return false;
	
	return false;
}


bool ElementChangeEventDemo::_onPostEdit(BPEntityChangeEventArgCR arg)
{
	vector<int> vct;
	BPViewManager::getInstance().getAllActiveViewports(vct);
	bool ismutiviews = vct.size() == 1 ? false : true;
	//如果不是多视口，就直接返回
	if (!ismutiviews)
		return false;
	BPDataKey datakey = arg.getDataKey();
	PModelId modid = arg.getModelId();
	BPEntityId entid = arg.getEntityId();
	BPProjectP project = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (project == NULL)
		return false;
	
	BPEntity entity(entid, *project);
	if (!entity.isValid())
		return false;
	::p3d::PString strClassName = L"";
	//多视口联动目前针对cube，其他对象就先不弄
	entity.getClassName(strClassName);
	if (strClassName != L"CubeDemo")
		return false;
	BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(entity);
	if (!ptrData.isValid())
		return false;

	DemoObject::CubeDemo pbCube;
	pbCube.initFromData(*ptrData);
	GeTransform trans = pbCube.getTransform();
	
	//获取三维cube绑定的二维entity
	set<BPEntityId> setids;
	BPDataUtil::getAllBindingEntityFromData(setids, datakey, project);
	
	BPViewportP secondview = BPViewManager::getInstance().getViewport(g_newview);
	if (secondview == nullptr)
		return false;
	p3d::pset<BPEntityId> all = BPViewController::GetAllElements(*secondview);
	for (BPEntityId id : setids/*all*/)
	{
		if(id == entid)//如果是对应的三维的那个entity，就continue
			continue;
		BPEntity ent(id, *BPProject::getActiveProject());
		if (ent.isValid())
		{
			ent.deleteFromModel();
		}
	}
	
	BPGraphicsPtr ptrGraps = pbCube.createGraphicsPlane(*project, g_ModelId);
	BPGraphicsUtils::transformPhysicalGraphics(*ptrGraps, trans);
	BPEntityId graid = ptrGraps->save();
	BPEntity graentity(graid, *project);
	if (!graentity.isValid())
		return false;

	P3DStatus sta = BPDataUtil::bindEntityToData(graid, datakey, project);
	if (sta != P3DStatus::SUCCESS)
		return false;
	return false;
}

bool ElementChangeEventDemo::_onPreDelete(BPEntityChangeEventArgCR arg)
{
	//返回true可阻止删除
	return false;
}
