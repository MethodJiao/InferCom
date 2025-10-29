#include "stdafx.h"

using namespace DemoObject;

void IndependentBridgePropertyDemo::OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)
{
	PBBimUIPropertyItem properties[IBPropName::enPROPCOUNT];
	vector<PBBimUIPropertyItem*> posItem;
	for (int i = 0; i < refps.size(); i++)
	{
		
		IBPObjectPtr ptrPbObj = BPObjectExtensionManager::getInstance().getBPObject(*refps.at(i));
		IndependentBridgePtr ptrIb = dynamic_cast<IndependentBridgeP>(ptrPbObj.get());
		if (!ptrIb.isValid())
			continue;

		{
			int nValueGet = ptrIb->getIBPattern();
			if (0 == i)
			{
				PBBimUIPropertyItem::ListValue listValue;
				listValue.m_strs.push_back(L"¹°Ê½ÇÅ¼Ü");
				listValue.m_strs.push_back(L"ÁºÊ½ÇÅ¼Ü");
				if (0 == nValueGet)
					listValue.m_sel = 0;
				else
					listValue.m_sel = 1;
				PBBimUIPropertyItem item(L"ÑùÊ½(P)", listValue);
				properties[IBPropName::enPATTERN].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enPATTERN];
				if (!oldItem->multiValue())
				{
					int nValueOld;
					oldItem->getValue(nValueOld);
					if (nValueOld != nValueGet)
						properties[IBPropName::enPATTERN].setMultiValue(true);
				}
			}
		}

		//ÇÅÖùÖ±¾¶
		{
			double dValueGet = ptrIb->getColumnDiameter();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅÖùÖ±¾¶(DZ)", dValueGet);
				properties[IBPropName::enCOLUMNDIAMETER].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enCOLUMNDIAMETER];
				if (!oldItem->multiValue())
				{
					double dvalueOld;
					oldItem->getValue(dvalueOld);
					if (abs(dvalueOld - dValueGet) > 0.01)
						properties[IBPropName::enCOLUMNDIAMETER].setMultiValue(true);
				}
			}
		}

		//ÇÅÖù¸ß
		{
			double dValueGet = ptrIb->getColumnHight();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅÖù¸ß(HZ)", dValueGet);
				properties[IBPropName::enCOLUMNHEIGHT].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enCOLUMNHEIGHT];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enCOLUMNHEIGHT].setMultiValue(true);
				}
			}
		}

		//ÇÅ¼Ü¿ç¾à
		{
			double dValueGet = ptrIb->getCSSLong();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅ¼Ü¿ç¾à(L)", dValueGet);
				properties[IBPropName::enCSSLENGHT].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enCSSLENGHT];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enCSSLENGHT].setMultiValue(true);
				}
			}
		}

		//ÇÅ¼Ü¿í
		{
			double dValueGet = ptrIb->getCSSWidth();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅ¼Ü¿í(W)", dValueGet);
				properties[IBPropName::enCSSWIDTH].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enCSSWIDTH];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enCSSWIDTH].setMultiValue(true);
				}
			}
		}

		//ÇÅ¼Ü¸ß
		{
			double dValueGet = ptrIb->getCSSHight();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅ¼Ü¸ß(H)", dValueGet);
				properties[IBPropName::enCSSHEIGHT].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enCSSHEIGHT];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enCSSHEIGHT].setMultiValue(true);
				}
			}
		}

		//ÇÅ¶¥°å¸ß
		{
			double dValueGet = ptrIb->getTopSlabThickness();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅ¶¥°å¸ß(H1)", dValueGet);
				properties[IBPropName::enTOPSLABTHICKNESS].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enTOPSLABTHICKNESS];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enTOPSLABTHICKNESS].setMultiValue(true);
				}
			}
		}

		//ÇÅ¹°¸ß
		if (0 == ptrIb->getIBPattern())
		{
			double dValueGet = ptrIb->getBridgeArchHight();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅ¹°¸ß(H2)", dValueGet);
				properties[IBPropName::enARCHHEIGHT].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enARCHHEIGHT];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enARCHHEIGHT].setMultiValue(true);
				}
			}
		}

		//ÇÅ±Úºñ
		{
			double dValueGet = ptrIb->getSideSlabThickness();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÇÅ±Úºñ(TQ)", dValueGet);
				properties[IBPropName::enSIDESLABTHICKNESS].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enSIDESLABTHICKNESS];
				if (!oldItem->multiValue())
				{
					double dValueOld;
					oldItem->getValue(dValueOld);
					if (abs(dValueOld - dValueGet) > 0.01)
						properties[IBPropName::enSIDESLABTHICKNESS].setMultiValue(true);
				}
			}
		}

		//ÅÅ¹ÜÊý
		{
			int nValueGet = ptrIb->getNumRows();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÅÅ¹ÜÐÐÊý(N)", nValueGet);
				properties[IBPropName::enROWS].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enROWS];
				if (!oldItem->multiValue())
				{
					int nValueOld;
					oldItem->getValue(nValueOld);
					if (nValueOld != nValueGet)
						properties[IBPropName::enROWS].setMultiValue(true);
				}
			}
		}

		//ÅÅ¹ÜÁÐÊý
		{
			int nValueGet = ptrIb->getNumColumns();
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÅÅ¹ÜÁÐÊý", nValueGet);
				properties[IBPropName::enCOLUMNS].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enCOLUMNS];
				if (!oldItem->multiValue())
				{
					int nValueOld;
					oldItem->getValue(nValueOld);
					if (nValueOld != nValueGet)
						properties[IBPropName::enCOLUMNS].setMultiValue(true);
				}
			}
		}

		//ÅÅ¹ÜÄÚ¾¶
		{
			vector<double> dias = ptrIb->getTubeDiameter();
			wstring value = __doubles2wstring(dias);

			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÅÅ¹ÜÄÚ¾¶(D1)", value.c_str());
				properties[IBPropName::enTUBEDIAMETER].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enTUBEDIAMETER];
				if (!oldItem->multiValue())
				{
					wstring valueOld;
					oldItem->getValue(valueOld);
					if (value != valueOld)
						properties[IBPropName::enTUBEDIAMETER].setMultiValue(true);
				}
			}
		}

		//ÅÅ¹Ü±Úºñ
		{
			vector<double> dDias = ptrIb->getTubeThickness();
			wstring value = __doubles2wstring(dDias);

			if (0 == i)
			{
				PBBimUIPropertyItem item(L"ÅÅ¹Ü±Úºñ(T1)", value.c_str());
				properties[IBPropName::enTUBETHICKNESS].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[IBPropName::enTUBETHICKNESS];
				if (!oldItem->multiValue())
				{
					wstring valueOld;
					oldItem->getValue(valueOld);
					if (value != valueOld)
						properties[IBPropName::enTUBETHICKNESS].setMultiValue(true);
				}
			}
		}

		//ÅÅ¹Ü×ø±ê
		pvector<GePoint3d> pts = ptrIb->getTubeCenters();
		int nPtsSize = pts.size();
		if (0 == i)
		{
			for (int j = 0; j < nPtsSize; j++)
			{
				wstring wsPosName = L"ÅÅ¹Ü¾Ö²¿×ø±ê(POS-";
				wsPosName += std::to_wstring(j + 1);
				wsPosName += L")";
				PBBimUIPropertyItem* item = new PBBimUIPropertyItem(wsPosName.c_str(), pts.at(j));
				posItem.push_back(item);
			}
		}
		else
		{
			if (nPtsSize != posItem.size())
			{
				for (auto ite : posItem)
				{
					if (ite)
					{
						delete ite;
						ite = NULL;
					}
				}
				posItem.clear();
				PBBimUIPropertyItem * item = new PBBimUIPropertyItem(L"ÅÅ¹Ü¾Ö²¿×ø±ê(POS)", 0);
				item->setMultiValue(true);
				posItem.push_back(item);
			}
			else
			{
				for (int j = 0; j < nPtsSize; j++)
				{
					GePoint3d oldValue;
					posItem.at(j)->getValue(oldValue);
					if (pts.at(j) != oldValue)
					{
						posItem.at(j)->setMultiValue(true);
					}
				}
			}
		}
				
	}

	lst.AppendGroup(L"¶ÀÁ¢ÇÅ¼ÜÊôÐÔ");

	int index = 0;
	for (int i = 0; i < IBPropName::enPROPCOUNT; ++i)
	{
		lst.Append(index++, properties[i]);
	}

	for (auto ite : posItem)
	{
		if (ite)
		{
			lst.Append(index++, *ite);
			delete ite;
			ite = NULL;
		}
	}
}

TIErrorStatus IndependentBridgePropertyDemo::OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item)
{
	BPProjectPtr ptrProject = NULL;
	IBPObjectPtr ptrPbObj = NULL;
	IndependentBridgePtr ptrIb = NULL;
	for (int i = 0; i < refps.size(); ++i)
	{
		ptrProject = refps.at(i)->getBPProject();
		if (ptrProject.isNull())
			continue;

		BPDataPtr ib = BPDataUtil::getDataOnEntity(*refps.at(i));
		if (!ib.isValid())
			continue;
		ptrIb = IndependentBridge::create();
	
		ptrIb->initFromData(*ib);

		if (ptrIb == nullptr)
			continue;

		switch (index)
		{
		case IBPropName::enPATTERN:
		{
			PBBimUIPropertyItem::ListValue value;
			item.getValue(value);
			ptrIb->setIBPattern((IBPattern)value.m_sel);
		}
		break;
		case  IBPropName::enCOLUMNDIAMETER:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setColumnDiameter(dValue);
		}
		break;
		case IBPropName::enCOLUMNHEIGHT:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setColumnHight(dValue);
		}
		break;
		case IBPropName::enARCHHEIGHT:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setBridgeArchHight(dValue);
		}
		break;
		case IBPropName::enROWS:
		{
			int nValue = 0;
			item.getValue(nValue);
			ptrIb->setNumRows(nValue);
		}
		break;
		case IBPropName::enCOLUMNS:
		{
			int nValue = 0;
			item.getValue(nValue);
			ptrIb->setNumColumns(nValue);
		}
		break;
		case IBPropName::enCSSLENGHT:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setCSSLong(dValue);
		}
		break;
		case IBPropName::enCSSWIDTH:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setCSSWidth(dValue);
		}
		break;
		case IBPropName::enCSSHEIGHT:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setCSSHight(dValue);
		}
		break;
		case IBPropName::enTOPSLABTHICKNESS:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setTopSlabThickness(dValue);
		}
		break;
		case IBPropName::enSIDESLABTHICKNESS:
		{
			double dValue = 0;
			item.getValue(dValue);
			ptrIb->setSideSlabThickness(dValue);
		}
		break;
		case IBPropName::enTUBEDIAMETER:
		{
			wstring value = L"";
			item.getValue(value);
			ptrIb->setTubeDiameter(__wstring2doubles(value));
		}
		break;
		case IBPropName::enTUBETHICKNESS:
		{
			wstring value = L"";
			item.getValue(value);
			ptrIb->setTubeThickness(__wstring2doubles(value));
		}
		break;


		default:
		{
			GePoint3d newPt = GePoint3d::createByZero();
			item.getValue(newPt);
			int ptIndex = index - (int)IBPropName::enPROPCOUNT;
			pvector<GePoint3d> pts = ptrIb->getTubeCenters();
			if (ptIndex>pts.size()-1)
				break;
			pts.at(ptIndex) = newPt;
			ptrIb->setTubeCenters(pts);

		}
			break;

		}
		ptrIb->replaceInProject(*ptrProject);

	}
	return TIErrorStatus::succeed;
}

vector<double> IndependentBridgePropertyDemo::__wstring2doubles(wstring ws)
{
	vector<double> nums;

	int nIndexSpace = ws.find_first_of(L" ");
	while (nIndexSpace != -1)
	{
		ws.erase(nIndexSpace, 1);
		nIndexSpace = ws.find_first_of(L" ");
	}

	int nIndexFullWidthComma = ws.find_first_of(L"£¬");
	while (nIndexFullWidthComma != -1)
	{
		ws.replace(nIndexFullWidthComma, 1, L",");
		nIndexFullWidthComma = ws.find_first_of(L"£¬");
	}

	int nIndex = ws.find_first_of(L",");
	while (nIndex  != -1)
	{
		nums.push_back(std::stod(ws.substr(0, nIndex)));
		ws = ws.substr(nIndex+1);
		nIndex = ws.find_first_of(L",");
	}
	nums.push_back(std::stod(ws));

	return nums;
}

std::wstring IndependentBridgePropertyDemo::__doubles2wstring(vector<double> doubs)
{
	wstring value = L"";
	wstring wsTem = L"";
	for (auto mem : doubs)
	{
		wsTem = std::to_wstring(mem);
		int nIndex = wsTem.find_first_of(L".");
		if (nIndex != -1)
			wsTem = wsTem.substr(0, nIndex + 3);
		value += wsTem;
		value += L",";
	}
	if (value.size() > 0)
		value.pop_back();
	return value;
}


class IndependentBridgePropertyDemoFactory :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		IndependentBridgePropertyDemo *p = new IndependentBridgePropertyDemo();
		p->AddRef();
		return p;
	}
};

static IndependentBridgePropertyDemoFactory s_IndependentBridgePropertyDemoFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory(PBM_CLASS_INDEPENDENT_BRIDGE, IToolNameProperty, &s_IndependentBridgePropertyDemoFactory);
AutoDoRegisterFunctionsEnd