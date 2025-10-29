#include "stdafx.h"
#include "ThreadDemo.h"



class ThreadFun : public BIMBase::IBPThreadJob
{

	BIMBase::IBPThreadJobP m_bpjob;
public:
	ThreadFun()
	{

	}

	virtual void _run(BIMBase::IBPThreadP) override
	{
		BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
		if (pProject == nullptr)
			return;
		BPModelBaseP pModel = pProject->getActiveModel();
		if (pModel == nullptr)
			return;
		BIMBase::Core::BPEntityArray entitys;
		::p3d::P3DStatus status = BIMBase::Core::BPEntityUtil::getEntitiesOfModel(entitys, *pProject, pModel->getModelId());
		//修改颜色和图层显示
		for (int i = 0; i < entitys.getCount(); i++)
		{
			BIMBase::Core::BPEntityPtr ptrEntity = entitys.getByIndex(i);
			if (!ptrEntity.isValid())
				continue;
			BIMBase::BPDataKey _key = BIMBase::Core::BPDataUtil::getDataKeyOnEntity(*ptrEntity);
			if (_key.isValid())
			{
				::BIMBase::PLayerId layerIdb;
				p3d::PString layerNames = _T("修改图层");
				Utf8String Ustr;

				P3DStringHelper::wCharToUtf8(Ustr, layerNames.c_str());

				p3d::P3DStatus sta = BIMBase::Core::BPLayerUtil::getLayerIdByName(layerIdb, *pProject, Ustr);
				BIMBase::Core::BPLayerInfo info;
				info.m_name = layerNames;
				info.m_lineWeight = 8;
				info.m_style = 0;
				info.m_isVisible = true;//这个要设置为true，不然图素在图层上不显示
				if (sta == P3DStatus::ERROR)//说明要新建一个这个图层
				{
					BIMBase::Core::BPLayerUtil::createLayer( layerIdb, *pProject, info);

				}
				ptrEntity->setLayerId(layerIdb);
				P3DStatus s = ptrEntity->replaceInModel(ptrEntity.get());

				//修改颜色
				BPGraphicsPtr ptrGraphic = BPGraphics::getGraphicsFromEntity(*ptrEntity.get());
				if (ptrGraphic.isNull())
					return;

				BIMBase::BPColorDef colorDef(0, 255, 0);

				UInt32 nColor = BPColorUtil::getEntityColor(colorDef, *pProject, true);
				UInt32 nWeight = 0, nColor2 = 0; Int32 nStyle = 0;

				BPSymbology sys;
				sys.color = nColor;
				sys.weight = nWeight;
				sys.style = nStyle;
				ptrGraphic->setSymbologySource(BPSymbologySource::enEntity);
				ptrGraphic->setSymbology(sys);
				ptrGraphic->updateEntityWithGraphics(ptrEntity.get());
			}
		}
	}

};

ThreadFunListener* ThreadFunListener::s_recordThread = nullptr;
ThreadFunListener& ThreadFunListener::Get()
{
	if (NULL == s_recordThread)
		s_recordThread = new ThreadFunListener();
	return *s_recordThread;
}

void  ThreadFunListener::BeginRecord()
{
	ThreadFun* job = new ThreadFun();
	m_threadP = job;
	BIMBase::IBPThreadPool::getThreadPool()->addJob(job);

}
void  ThreadFunListener::EndRecord()
{
	if (m_threadP)
	{
		BIMBase::IBPThreadPool::getThreadPool()->waitJobFinished();
		delete m_threadP;
		m_threadP = nullptr;
	}
}

void testThreadDemo()
{


	ThreadFunListener::Get().BeginRecord();
	ThreadFunListener::Get().EndRecord();

}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("ThreadDemo", &testThreadDemo);
AutoDoRegisterFunctionsEnd