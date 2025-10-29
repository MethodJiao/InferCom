#include "stdafx.h"
#include "MyOpenningPropertyDemo.h"
#include "OpenningDemo.h"

using namespace DemoObject;
MyOpenningPropertyDemo::MyOpenningPropertyDemo()
{
}


MyOpenningPropertyDemo::~MyOpenningPropertyDemo()
{
}

void MyOpenningPropertyDemo::OnPropertyGet(std::vector<BPEntityP> const & refps, PBBimUIProperyList& lst)
{	
	//选择集中的对象的个数
	if (refps.size() == 0)
		return;	
	BPEntityId eleId;
	std::vector<IBPObjectPtr> pbObjs;
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	
	for (auto elementRef : refps)
	{
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(*elementRef);
		if (!ptrPbObj.isValid())
			continue;
		pbObjs.push_back(ptrPbObj);
	}	
	
	if (pbObjs.size() < 1)
		return;

	PBBimUIPropertyItem properties[OpenningPropName::OpenningPropCount];
	for (int i = 0; i < pbObjs.size(); ++i)
	{
		IBPObjectPtr objectPtr = pbObjs.at(i);
		OpenningDemoP pOpenning = dynamic_cast<OpenningDemoP>(objectPtr.get());
		if (NULL == pOpenning)
			continue;

		// ----------------------------------------长度---------------------------------------
		{
			int nLength = pOpenning->getLength();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"长度", nLength);
				properties[OpenningPropName::Length].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[OpenningPropName::Length];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int oldLength;
					oldItem->getValue(oldLength);
					if (oldLength != nLength)
						properties[OpenningPropName::Length].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------高度---------------------------------------
		{
			int nHeight = pOpenning->getHeight();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"高度", nHeight);
				properties[OpenningPropName::Height].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[OpenningPropName::Height];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int noldHeight;
					oldItem->getValue(noldHeight);
					if (noldHeight != nHeight)
						properties[OpenningPropName::Height].setMultiValue(true);
				}
			}
		}
	}

	lst.AppendGroup(L"构件属性");	
	int nIndex = 0;
	for (int i = 0; i < OpenningPropName::OpenningPropCount; ++i)
	{
		lst.Append(nIndex++, properties[i]);
	}
}

TIErrorStatus MyOpenningPropertyDemo::OnPropertySet(std::vector<BPEntityP> const & refps, int index, PBBimUIPropertyItem const & item)
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return TIErrorStatus::failed;

	std::vector<IBPObjectPtr> pbObjs;
	for (auto elementRef : refps)
	{
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(*elementRef);
		if (!ptrPbObj.isValid())
			continue;
		pbObjs.push_back(ptrPbObj);
	}

	if (pbObjs.size() < 1)
		return TIErrorStatus::failed;

	for (int indexGj = 0; indexGj < pbObjs.size(); ++indexGj)
	{
		IBPObjectPtr ptrObject = pbObjs.at(indexGj);
		OpenningDemoP pOpenning = dynamic_cast<OpenningDemo*>(ptrObject.get());
		if (pOpenning == nullptr)
			continue;

		switch (index)
		{
		case OpenningPropName::Length:
		{
			int nLen = 0;
			item.getValue(nLen);
			pOpenning->setLength(nLen);
		}
		break;
		case OpenningPropName::Height:
		{
			int nHeight = 0;
			item.getValue(nHeight);
			pOpenning->setHeight(nHeight);
		}
		break;
		default:
			break;

		}
		pOpenning->replaceInProject(*pProject);
		BPDataKeyArray relDataIds;
		BPRelationshipFinder::getRelatedDatasByRelationship(relDataIds, pOpenning->getDataKey(), *pProject, PBM_SCHEMA_Demo, PBM_RELSHIP_CUBEWITHOPENNING);
		for each (auto dataKey in relDataIds)
		{
			BPDataPtr ptrData = BPDataUtil::getDataByKey(dataKey, *pProject);
			if (ptrData == nullptr)
				continue;
			CubeDemoPtr ptrCube = CubeDemo::create(*ptrData);
			if (ptrCube == nullptr)
				continue;

			ptrCube->replaceGraphics(*pProject);
		}
	}
	return TIErrorStatus::succeed;
}


class OpenningPropertyFactoryDemo :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		MyOpenningPropertyDemo *p = new MyOpenningPropertyDemo();
		p->AddRef();
		return p;
	}
};

static OpenningPropertyFactoryDemo s_OpenningPropertyFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("OpenningDemo", IToolNameProperty, &s_OpenningPropertyFactory);
AutoDoRegisterFunctionsEnd