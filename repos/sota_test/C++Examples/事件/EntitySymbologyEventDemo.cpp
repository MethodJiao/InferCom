#include "stdafx.h"
#include "EntitySymbologyEventDemo.h"


using namespace PBBim::PBCD;

ClashMethod::ClashMethod()
{
}


ClashMethod::~ClashMethod()
{
	for (auto node : m_vecTrasNode)
	{
		delete node;
		node = nullptr;
	}
}

bool ClashMethod::doClash(ClashRule const& rule)
{
	m_rule = rule;
	__preFilter();
	__runClashDetection();
	return true;
}

void ClashMethod::getClashResult(ClashResult& result)
{
	result = m_clashResult;
}

void ClashMethod::__preFilter()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	m_clashResult.clear();
	m_vecClashEle.clear();
	m_vecTrasNode.clear();
	for (auto entity : m_rule.m_vecEntity)
	{
		CDObjectNode* node = new CDObjectNode();
		CDAABB aabbRange;
		BPModelP pModel = entity->getBPModel();
		if (pModel == nullptr)
			continue;
		PModelId modleId = pModel->getModelId();
		auto iter = m_rule.m_mapModelTransform.find(modleId);
		if (iter != m_rule.m_mapModelTransform.end())
		{
			GeTransform modelTrans = iter->second;
			BPGraphicsPtr ptrGraphic = BPEntityUtil::transformEntity(*entity, modelTrans, false);
			if (ptrGraphic.isNull())
				continue;
			GeRange3d range;
			BPGraphicsUtils::getRangeOfPhysicalGraphics(range,*ptrGraphic);
			node->range.ptMin = AcGePoint3d(range.low.x, range.low.y, range.low.z);
			node->range.ptMax = AcGePoint3d(range.high.x, range.high.y, range.high.z);

			CDElementFunction::AddPhysicalGraphicsToCDBvVector(ptrGraphic, node->vctReal);//真实几何
			CDElementFunction::AddPhysicalGraphicsToCDBvVector(ptrGraphic, node->vctBv);//真实几何代替OBB
			m_vecClashEle.push_back(entity);
			m_vecTrasNode.push_back(node);
		}
		else
		{
			CDElementFunction::GetElementRange(*pProject, entity->getEntityId(), aabbRange);
			node->range = aabbRange;
			BPEntityPtr enti = pModel->findEntityByID(entity->getEntityId());
			if (enti.isNull())
				continue;
			BPGraphicsPtr ptrGraphic = BPGraphics::getGraphicsFromEntity(*enti);
			//BPGraphicsPtr ptrGraphic = pModel->ReadPhysicalGraphics(entity->getEntityId());
			if (ptrGraphic.isNull())
				continue;
			CDElementFunction::AddPhysicalGraphicsToCDBvVector(ptrGraphic, node->vctReal);//真实几何
			CDElementFunction::AddPhysicalGraphicsToCDBvVector(ptrGraphic, node->vctBv);//真实几何代替OBB
			m_vecClashEle.push_back(entity);
			m_vecTrasNode.push_back(node);
		}
	}
}

void ClashMethod::__runClashDetection()
{
	m_clashResult.clear();
	for (auto iter = m_vecClashEle.begin(); iter != m_vecClashEle.end(); iter++)
	{
		for (auto iterRight = m_vecClashEle.begin(); iterRight != m_vecClashEle.end(); iterRight++)
		{
			if (iter == iterRight)
				continue;
			int nLeft = iter - m_vecClashEle.begin();
			int nRight = iterRight - m_vecClashEle.begin();
			if (nLeft > nRight)
				continue;
			CDObjectNode* pNodeLeft = m_vecTrasNode[nLeft];
			CDObjectNode* pNodeRight = m_vecTrasNode[nRight];
			AcGePoint3d point;
			bool bClash = pNodeLeft->IntersectWith(pNodeRight, point);
			if (bClash)
			{
				m_clashResult.push_back(make_pair(*iter, *iterRight));
			}
		}
	}
}


void EntitySymbologyEventDemo::_getOverrides(BPSymbologyOverridesR overrids, ::BIMBase::Core::BPEntityCR eh) const
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	if (!eh.isValid())
		return;
	BPEntityId entityId = eh.getEntityId();
	bool bIsOveride = false;
	//遍历选择集
	if (m_selected.find(entityId) == m_selected.end())
		return;
	overrids.setOveride(true);
	COLORREF oldColor = RGB(233, 184, 119);
	overrids.setColor(oldColor);
	overrids.setTransparency(0.9);
}

EntitySymbologyEventDemo& EntitySymbologyEventDemo::Get()
{
	static EntitySymbologyEventDemo event;
	return event;
}

void EntitySymbologyEventDemo::begin()
{
	if (!m_bHaveRegisted)
	{
		BPEntitySymbologyEventListenerCenter::getInstance().addListener(this);
		m_bHaveRegisted = true;
	}
}

void EntitySymbologyEventDemo::end()
{
	if (m_bHaveRegisted)
	{
		BPEntitySymbologyEventListenerCenter::getInstance().dropListener(this);
		m_bHaveRegisted = false;
	}
}

void EntitySymbologyEventDemo::setSelected(set<BPEntityId>& result)
{
	cs.Lock();
	m_selected.clear();
	m_selected = result;
	cs.Unlock();
}
