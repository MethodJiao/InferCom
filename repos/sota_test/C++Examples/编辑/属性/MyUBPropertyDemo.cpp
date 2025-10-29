#include "stdafx.h"
#include "MyUBPropertyDemo.h"
#include "UniversalBeamDemo.h"

using namespace DemoObject;
MyUBPropertyDemo::MyUBPropertyDemo()
{
}


MyUBPropertyDemo::~MyUBPropertyDemo()
{
}

void MyUBPropertyDemo::OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)
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
	PBBimUIPropertyItem properties[UBPropName::UBPropCount];
	for (int i = 0; i < pbObjs.size(); ++i)
	{
		IBPObjectPtr ptrObject = pbObjs.at(i);
		UniversalBeamDemoP pUB = dynamic_cast<UniversalBeamDemoP>(ptrObject.get());
		if (NULL == pUB)
			continue;

		// ----------------------------------------长度---------------------------------------
		{
			int nLength = pUB->getLength();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"长度", nLength);
				properties[UBPropName::Length].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[UBPropName::Length];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int oldLength;
					oldItem->getValue(oldLength);
					if (oldLength != nLength)
						properties[UBPropName::Length].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------厚度---------------------------------------
		{
			int nThick = pUB->getThick();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"厚度", nThick);
				properties[UBPropName::Thick].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[UBPropName::Thick];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int nNoldThick;
					oldItem->getValue(nNoldThick);
					if (nNoldThick != nThick)
						properties[UBPropName::Thick].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------宽度---------------------------------------
		{
			int nWidth = pUB->getWidth();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"宽度", nWidth);
				properties[UBPropName::Width].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[UBPropName::Width];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int nOldWidth;
					oldItem->getValue(nOldWidth);
					if (nOldWidth != nWidth)
						properties[UBPropName::Width].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------里层高度---------------------------------------
		{
			int nInnerHeight = pUB->getInnerHeight();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"里层高度", nInnerHeight);
				properties[UBPropName::InnerHeight].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[UBPropName::InnerHeight];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int nOldInnerHeight;
					oldItem->getValue(nOldInnerHeight);
					if (nOldInnerHeight != nInnerHeight)
						properties[UBPropName::InnerHeight].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------扩展属性---------------------------------------
		{
			BPExtendPropertySet extendPropertySet = pUB->getExtendPropertySet();
			BPPropertyValue propertyValue;
			CString sExtendValue = _T("");
			if (extendPropertySet.getSubParam(_T("扩展属性"), propertyValue))
			{
				if (propertyValue.m_type = BPPropertyValue::String)
				{
					sExtendValue = propertyValue.m_value.m_valueString.c_str();
				}
			}

			if (0 == i)
			{
				PBBimUIPropertyItem item(L"扩展属性", sExtendValue);
				properties[UBPropName::ExtendProperty].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[UBPropName::ExtendProperty];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					std::wstring oldExtendValue;
					oldItem->getValue(oldExtendValue);
					if (oldExtendValue != sExtendValue.GetString())
						properties[UBPropName::ExtendProperty].setMultiValue(true);
				}
			}
		}

	}

	lst.AppendGroup(L"构件属性");
	int nIndex = 0;

	for (int i = 0; i < UBPropName::UBPropCount; ++i)
	{
		lst.Append(nIndex++, properties[i]);
	}

}

TIErrorStatus MyUBPropertyDemo::OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item)
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
		UniversalBeamDemoP pUB = dynamic_cast<UniversalBeamDemo*>(ptrObject.get());
		if (pUB == nullptr)
			continue;

		switch (index)
		{
		case UBPropName::Length:
		{
			int nLen = 0;
			item.getValue(nLen);
			pUB->setLength(nLen);
		}
		break;
		case UBPropName::Thick:
		{
			int nThick = 0;
			item.getValue(nThick);
			pUB->setThick(nThick);
		}
		break;
		case UBPropName::InnerHeight:
		{
			int nHeight = 0;
			item.getValue(nHeight);
			pUB->setInnerHeight(nHeight);
		}
		break;
		case UBPropName::Width:
		{
			int nWidth = 0.0;
			item.getValue(nWidth);
			pUB->setWidth(nWidth);
		}
		break;
		default:
			break;

		}
		pUB->replaceInProject(*pProject);
	}
	return TIErrorStatus::succeed;
}


class UBPropertyFactoryDemo :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		MyUBPropertyDemo* p = new MyUBPropertyDemo();
		p->AddRef();
		return p;
	}
};

static UBPropertyFactoryDemo s_UBPropertyFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("UniversalBeamDemo", IToolNameProperty, &s_UBPropertyFactory);
AutoDoRegisterFunctionsEnd