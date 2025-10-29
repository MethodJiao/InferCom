#include "stdafx.h"
#include "MyCubePropertyDemo.h"
#include "CubeDemo.h"

using namespace DemoObject;
MyCubePropertyDemo::MyCubePropertyDemo()
{
}


MyCubePropertyDemo::~MyCubePropertyDemo()
{
}

void MyCubePropertyDemo::OnPropertyGet(std::vector<BPEntityP> const & refps, PBBimUIProperyList& lst)
{	
	//选择集中的对象的个数
	if (refps.size() == 0)
		return;	
	BPEntityId eleId;
	std::vector<IBPObjectPtr> pbObjs;
	BPProjectP pProject = nullptr;	
	
	for (auto elementRef : refps)
	{
		if (NULL == pProject)
		{
			pProject = elementRef->getBPProject();
			if (pProject == NULL)
				continue;
		}
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(*elementRef);
		if (!ptrPbObj.isValid())
			continue;
		pbObjs.push_back(ptrPbObj);
	}	
	
	if (pbObjs.size() < 1)
		return;
	PBBimUIPropertyItem properties[CubePropName::CubePropCount];
	for (int i = 0; i < pbObjs.size(); ++i)
	{
		IBPObjectPtr ptrObject = pbObjs.at(i);
		CubeDemoP pCube = dynamic_cast<CubeDemoP>(ptrObject.get());
		if (NULL == pCube)
			continue;

		// ----------------------------------------长度---------------------------------------
		{
			int nLength = pCube->getLength();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"长度", nLength);
				properties[CubePropName::Length].swap(item);
			}
			else
			{
				PBBimUIPropertyItem *oldItem = &properties[CubePropName::Length];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int oldLength;
					oldItem->getValue(oldLength);
					if (oldLength != nLength)
						properties[CubePropName::Length].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------高度---------------------------------------
		{
			int nHeight = pCube->getHeight();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"高度", nHeight);
				properties[CubePropName::Height].swap(item);
			}
			else
			{
				PBBimUIPropertyItem *oldItem = &properties[CubePropName::Height];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int nNoldHeight;
					oldItem->getValue(nNoldHeight);
					if (nNoldHeight != nHeight)
						properties[CubePropName::Height].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------宽度---------------------------------------
		{
			int nWidth = pCube->getWidth();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"宽度", nWidth);
				properties[CubePropName::Width].swap(item);
			}
			else
			{
				PBBimUIPropertyItem *oldItem = &properties[CubePropName::Width];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int nOldWidth;
					oldItem->getValue(nOldWidth);
					if (nOldWidth != nWidth)
						properties[CubePropName::Width].setMultiValue(true);
				}
			}
		}
		
		// ----------------------------------------扩展属性---------------------------------------
		{
			BPExtendPropertySet extendPropertySet = pCube->getExtendPropertySet();
			BPPropertyValue propertyValue;
			CString sExtendValue = _T("");
			if (extendPropertySet.getSubParam(_T("扩展属性"), propertyValue))
			{
				if (propertyValue.m_type = BPPropertyValue::String)
				{
					sExtendValue = propertyValue.m_value.m_valueString.c_str();
				}
			}
			else if (extendPropertySet.getSubParam(_T("墙自定义属性"), propertyValue))
			{
				if (propertyValue.m_type = BPPropertyValue::String)
				{
					sExtendValue = propertyValue.m_value.m_valueString.c_str();
				}
			}

			if (0 == i)
			{
				PBBimUIPropertyItem item(L"扩展属性", sExtendValue);
				properties[CubePropName::ExtendProperty].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[CubePropName::ExtendProperty];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					std::wstring oldExtendValue;
					oldItem->getValue(oldExtendValue);
					if (oldExtendValue != sExtendValue.GetString())
						properties[CubePropName::ExtendProperty].setMultiValue(true);
				}
			}
		}
	
	}
	
	lst.AppendGroup(L"构件属性");	
	int nIndex = 0;
	
	for (int i = 0; i < CubePropName::CubePropCount ; ++i)
	{
		lst.Append(nIndex++, properties[i]);
	}
	
}

TIErrorStatus MyCubePropertyDemo::OnPropertySet(std::vector<BPEntityP> const & refps, int index, PBBimUIPropertyItem const & item)
{
	BPProjectP pProject = NULL;
	std::vector<IBPObjectPtr> pbObjs;
	for (auto elementRef : refps)
	{
		if (NULL == pProject)
		{
			pProject = elementRef->getBPProject();
			if (pProject == NULL)
				continue;
		}
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
		CubeDemoP pCube = dynamic_cast<CubeDemo*>(ptrObject.get());
		if (pCube == nullptr)
			continue;

		switch (index)
		{
		case CubePropName::Length:
		{
			int nLen = 0;
			item.getValue(nLen);
			pCube->setLength(nLen);
		}
		break;
		case CubePropName::Height:
		{
			int nHeight = 0;
			item.getValue(nHeight);
			pCube->setHeight(nHeight);
		}
		break;
		case CubePropName::Width:
		{
			int nWidth = 0.0;
			item.getValue(nWidth);
			pCube->setWidth(nWidth);
		}
		break;		
		default:
			break;

		}
		pCube->replaceInProject(*pProject);

	}
	return TIErrorStatus::succeed;
}


class CubePropertyFactoryDemo :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		MyCubePropertyDemo *p = new MyCubePropertyDemo();
		p->AddRef();
		return p;
	}
};

static CubePropertyFactoryDemo s_CubePropertyFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("CubeDemo", IToolNameProperty, &s_CubePropertyFactory);
AutoDoRegisterFunctionsEnd