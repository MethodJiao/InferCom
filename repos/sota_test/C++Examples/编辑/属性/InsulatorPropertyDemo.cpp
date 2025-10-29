#include "stdafx.h"
#include "InsulatorPropertyDemo.h"
#include "InsulatorDemo.h"



using namespace DemoObject;
InsulatorPropertyDemo::InsulatorPropertyDemo()
{
}


InsulatorPropertyDemo::~InsulatorPropertyDemo()
{
}

void InsulatorPropertyDemo::OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)
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
	PBBimUIPropertyItem properties[InsulatorPropName::InsulatorPropCount];
	for (int i = 0; i < pbObjs.size(); ++i)
	{
		IBPObjectPtr ptrObject = pbObjs.at(i);
		InsulatorDemoP pInsulator = dynamic_cast<InsulatorDemoP>(ptrObject.get());
		if (NULL == pInsulator)
			continue;

		/**联数*/
		{
			int nN = pInsulator->getN();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"联数", nN);
				properties[InsulatorPropName::N].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::N];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int oldN;
					oldItem->getValue(oldN);
					if (oldN != nN)
						properties[InsulatorPropName::N].setMultiValue(true);
				}
			}
		}

		/**单串绝缘子片数量*/
		{
			int nN1 = pInsulator->getN1();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"单串绝缘子片数量", nN1);
				properties[InsulatorPropName::N1].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::N1];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					int oldN1;
					oldItem->getValue(oldN1);
					if (oldN1 != nN1)
						properties[InsulatorPropName::N1].setMultiValue(true);
				}
			}
		}

		/**绝缘子单片连接高度*/
		{
			double dH1 = pInsulator->getH1();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"绝缘子单片连接高度", dH1);
				properties[InsulatorPropName::H1].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::H1];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldH1;
					oldItem->getValue(oldH1);
					if (oldH1 != dH1)
						properties[InsulatorPropName::H1].setMultiValue(true);
				}
			}
		}

		/**大伞裙半径*/
		{
			double dR1 = pInsulator->getR1();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"大伞裙半径", dR1);
				properties[InsulatorPropName::R1].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::R1];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldR1;
					oldItem->getValue(oldR1);
					if (oldR1 != dR1)
						properties[InsulatorPropName::N1].setMultiValue(true);
				}
			}
		}

		/**小伞裙半径*/
		{
			double dR2 = pInsulator->getR2();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"小伞裙半径", dR2);
				properties[InsulatorPropName::R2].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::R2];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldR2;
					oldItem->getValue(oldR2);
					if (oldR2 != dR2)
						properties[InsulatorPropName::R2].setMultiValue(true);
				}
			}
		}

		/**绝缘子串半径*/
		{
			double dR = pInsulator->getR();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"绝缘子串半径", dR);
				properties[InsulatorPropName::R].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::R];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldR;
					oldItem->getValue(oldR);
					if (oldR != dR)
						properties[InsulatorPropName::R].setMultiValue(true);
				}
			}
		}


		/**双串间距*/
		{
			double dD = pInsulator->getD();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"双串间距", dD);
				properties[InsulatorPropName::D].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::D];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldD;
					oldItem->getValue(oldD);
					if (oldD != dD)
						properties[InsulatorPropName::D].setMultiValue(true);
				}
			}
		}


		/**前端长度（构架端*/
		{
			double dFL = pInsulator->getFL();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"构架端长度", dFL);
				properties[InsulatorPropName::FL].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::FL];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldFL;
					oldItem->getValue(oldFL);
					if (oldFL != dFL)
						properties[InsulatorPropName::FL].setMultiValue(true);
				}
			}
		}

		/**后端长度（导线端*/
		{
			double dAL = pInsulator->getAL();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"导线端长度", dAL);
				properties[InsulatorPropName::AL].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[InsulatorPropName::AL];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldAL;
					oldItem->getValue(oldAL);
					if (oldAL != dAL)
						properties[InsulatorPropName::AL].setMultiValue(true);
				}
			}
		}

	}

	lst.AppendGroup(L"构件属性");
	int nIndex = 0;

	for (int i = 0; i < InsulatorPropName::InsulatorPropCount; ++i)
	{
		lst.Append(nIndex++, properties[i]);
	}

}

TIErrorStatus InsulatorPropertyDemo::OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item)
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
		InsulatorDemoP pInsulator = dynamic_cast<InsulatorDemo*>(ptrObject.get());
		if (pInsulator == nullptr)
			continue;

		switch (index)
		{
		case InsulatorPropName::N:
		{
			int nN = 0;
			item.getValue(nN);
			pInsulator->setN(nN);
		}
		break;
		case InsulatorPropName::N1:
		{
			int nN1 = 0;
			item.getValue(nN1);
			pInsulator->setN1(nN1);
		}
		break;
		case InsulatorPropName::H1:
		{
			double nH1 = 0.0;
			item.getValue(nH1);
			pInsulator->setH1(nH1);
		}
		break;
		case InsulatorPropName::R1:
		{
			double dR1 = 0.0;
			item.getValue(dR1);
			pInsulator->setR1(dR1);
		}
		break;
		case InsulatorPropName::R2:
		{
			double dR2 = 0.0;
			item.getValue(dR2);
			pInsulator->setR2(dR2);
		}
		break;
		case InsulatorPropName::R:
		{
			double dR = 0.0;
			item.getValue(dR);
			pInsulator->setR(dR);
		}
		break;
		case InsulatorPropName::D:
		{
			double dD = 0.0;
			item.getValue(dD);
			pInsulator->setD(dD);
		}
		break;
		case InsulatorPropName::FL:
		{
			double dFL = 0.0;
			item.getValue(dFL);
			pInsulator->setFL(dFL);
		}
		break;
		case InsulatorPropName::AL:
		{
			double dAL = 0.0;
			item.getValue(dAL);
			pInsulator->setAL(dAL);
		}
		break;

		default:
			break;

		}
		pInsulator->replaceInProject(*pProject);

	}
	return TIErrorStatus::succeed;
}


class InsulatorPropertyFactoryDemo :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		InsulatorPropertyDemo* p = new InsulatorPropertyDemo();
		p->AddRef();
		return p;
	}
};

static InsulatorPropertyFactoryDemo s_InsulatorPropertyFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("InsulatorDemo", IToolNameProperty, &s_InsulatorPropertyFactory);
AutoDoRegisterFunctionsEnd