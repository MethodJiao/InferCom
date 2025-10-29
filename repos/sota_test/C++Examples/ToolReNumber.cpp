#include "stdafx.h"
#include "ToolReNumber.h"
#include "ToolCreateTextTest.h"

using namespace PBBim;
using namespace PBBim::PBBimCore;
using namespace ::p3d::platform;

ToolReNumber::ToolReNumber()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	
}


ToolReNumber::~ToolReNumber()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->Clear();
	//TOOLSETTINGMODELESSDLG_CLOSE;
}

void ToolReNumber::_onPostInstall()
{
	T_Super::_onPostInstall();
	//TOOLSETTING_OPEN;
	//_buildAgenda(NULL);
	//ElemSource es = _getElemSource();
	//_setLocateCursor(true);
	BPSnap::getInstance().enableSnap(true);
	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"请选择需要重新编号的类型").c_str());
}


void ToolReNumber::_setupAndPromptForNextAction()
{
	/*if (SOURCE_Pick != _getElemSource())
		return;*/
}

void   ToolReNumber::_onRestartTool()
{
	ToolReNumber* newTool = new ToolReNumber();
	newTool->installTool();
}

bool ToolReNumber::_onDataButton( BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);
	PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"右键确定重新编号").c_str());
	return true;
}

void ToolReNumber::_onDynamicFrame( BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;
}

::BPEntityPtr ToolReNumber::_buildLocateAgenda(BPPickDataCP path, BPBaseButtonEventCP ev)
{
	::BPEntityPtr eh = T_Super::_buildLocateAgenda(path, ev);
	//     PDEditEntityHandle eeh;
	//     eeh.Duplicate(*P3DInnerFriend::getInnerEntityP(*eh->GetElementRef()));
	m_vcEEH.push_back(eh);
	return eh;
}

bool ToolReNumber::_onResetButton( BPBaseButtonEventCP ev)
{
	// 获取当前工程
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return false;
	BPProjectPtr ptrProject = pProjectManager->getMainProject();
	if (ptrProject.isNull())
		return false;
	BPModelBaseP model = ptrProject->getActiveModel();
	if (model == NULL) {
		return false;
	}
	::BIMBase::PModelId modelId = model->getModelId();
	if (!modelId.IsValid()) {
		return false;
	}

	PBModelInfoPtr modelInfo = PBModelInfoManager::Get().GetModelById(modelId);
	if (!modelInfo.isValid()){
		return false;
	}
	// 将id转化为BPDataKey(为组件函数所需要的数据类型)
	pvector<BPDataKey> vctKeys;
	for (size_t i = 0; i < m_vcEEH.size(); i++)
	{
		BPEntityP entity = m_vcEEH[i].get();
		if (entity == nullptr)
			continue;
		if (entity->isValid())
		{
			// 从实例单元中拿取数据
			BPDataKey key=BPDataUtil::getDataKeyOnEntity(*entity);
			if (key.isValid())
			{
				vctKeys.push_back(key);
			}
		}
	}
	bool res = false;
	m_vcEEH.clear();
	getEntityArray()->clear();
	if (vctKeys.size() != 1)
		return false;

	BPDataPtr ptrInstance = BPDataUtil::getDataByKey(vctKeys[0], *ptrProject);
	IBPObjectPtr ptrI = BPObjectExtensionManager::getInstance().getBPObject(*ptrInstance);
	if (ptrI.isNull())
		return false;
	BPComponentElementPtr ptrObj = dynamic_cast<BPComponentElementP>(ptrI.get());
	if (ptrObj.isNull())
		return false;
	BPComponentPrototypePtr  ptrPrototype = BPComponentPrototype::create();
	if (!ptrObj->getComponentPrototype(*ptrPrototype, *ptrProject))
		return false;

	/*PString str = L"手车闸刀";
	BPComponentPrototypePtr ptrPrototype = BPComponentManager::getInstance(project).getPrototypeByName(str);
	if (ptrPrototype.isNull())
		return false;*/

	CString strName = _T("编号");
	BPComItemProp nameProperty;
	if (ptrPrototype->getPropItem(L"类型1", L"编号名称", nameProperty))
	{
		strName = nameProperty.psItemVal.at(0).c_str();
	}
	else
	{
		//为组件添加类型属性
		BPComItemProp property;
		property.psItemName = L"编号名称";
		property.psItemVal = vector<PString>{L"编号"};
		property.eCtrlType = BPComCtrlType::enEdit;
		property.eItemValType = BPComValType::enString;
		property.bIsReadOnly = false;
		ptrPrototype->addTypeProperty(property);
		//为组件添加实例属性
		BPComItemProp property2;
		property2.psItemName = L"编号";
		property2.psItemVal = vector<PString>{L"0"};
		property2.eCtrlType = BPComCtrlType::enEdit;
		property2.eItemValType = BPComValType::enInt;
		property2.bIsReadOnly = false;
		ptrPrototype->addElemProperty(property2);
		ptrPrototype->updateProperty(*ptrProject);
		ptrPrototype->replaceData(*ptrProject);
	}

	pvector<BPGraphicElementPtr> vctElement;
	if (!ptrPrototype->getAllComponentElement(vctElement, *ptrProject))
		return false;	

	auto f = [](BPGraphicElementPtr r, BPGraphicElementPtr l)
	{
		GePoint3d pointr = r->getPlacement().getOrigin();
		double xr = pointr.x;
		double yr = pointr.y;
		GePoint3d pointl = l->getPlacement().getOrigin();
		double xl = pointl.x;
		double yl = pointl.y;
		if (xr < xl)
			return true;
		else if (abs(xr-xl) < 0.01)
		{
			if (yr < yl)
				return true;
			else
				return false;
		}
		else
			return false;
	};
	std::sort(vctElement.begin(), vctElement.end(), f);
	
	int nCount = 1;
	for (BPGraphicElementPtr ele : vctElement)
	{
		if (ele.isNull())
			continue;

		BPComponentElementPtr ptrEle = dynamic_cast<BPComponentElementP>(ele.get());
		if (ptrEle.isNull())
			return true;

		BPEntityPtr ee;
		(ele->getData(*ptrProject))->getElement(ee);
		if (ee.isNull())
			continue;

		GeRange3d range = GeRange3d::createByNull();
		ee->getRange(range);
		GePoint3d pt = range.high;

		CString strNum, str;
		strNum.AppendFormat(_T("%d"), nCount);
		str = strName + strNum;


		//判断是否已经编号
		BPDataKey keyGroup = BPGroupUtil::getGroupKeyBySubKey(ele->getDataKey(), project);
		if (keyGroup.isValid())
		{
			vector<BPDataKey> vctKes;
			BPGroupUtil::getSubKeysByGroupKey(keyGroup, vctKes, ptrProject);
			BPDataKey keyText;
			for (auto key : vctKes)
			{
				if (key == ele->getDataKey())
					continue;
				else
					keyText = key;
			}

			BPDataPtr ptrInstanceText = BPDataUtil::getDataByKey(keyText, *ptrProject);
			if (ptrInstanceText.isNull())
				continue;	
			//BPGroupUtil::DisGroup(keyGroup);
			IBPObjectPtr ptrIText = BPObjectExtensionManager::getInstance().getBPObject(*ptrInstanceText);
			if (ptrIText.isNull())
				continue;
			BPTextEntityPtr ptrText = dynamic_cast<BPTextEntityP>(ptrIText.get());
			if (ptrText.isNull())
				continue;
			ptrText->setContent(str.GetString());
			ptrText->replaceInProject(*ptrProject);
			/*BPDataKey keyText2 = ptrText->getDataKey();

			KeyVct vctKey2;
			vctKey2.push_back(ptrEle->getDataKey());
			vctKey2.push_back(keyText2);

			BPDataKey kk = BPGroupUtil::group(vctKey2, str);*/
		}
		else
		{
			BPTextEntityPtr text = ToolCreateTextTest::createText2(str.GetString(), pt);//ToolCreateTextTest::createText(str.GetString(), pt);
			if (text.isNull())
				continue;			
			
			map<PString, BPComItemProp> mapProp;
			if (!ptrEle->getElemPropItems(mapProp))
				continue;
			if (mapProp.find(L"编号") != mapProp.end())
			{
				mapProp.find(L"编号")->second.psItemVal = vector<PString>{strNum.GetString()};
			}
			ptrEle->setElemPropItems(mapProp);
			ptrEle->replaceData(*ptrProject);

			BPDataKey keyText = text->getDataKey();

			KeyVct vctKey;
			vctKey.push_back(ptrEle->getDataKey());
			vctKey.push_back(keyText);

			BPDataKey keyGroup = BPGroupUtil::group(vctKey, str);
		}

		nCount++;
	}
	

	return true;
}

bool ToolReNumber::_onKeyTransition(bool wentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)
{
	return true;
}

/*
void ToolReNumber::_OnReceive(Utf8CP messageType, JsonValueCR messageDataObj)
{
	TOOLSETTING_CHECK;	

}*/

/*
p3d::StatusInt ToolReNumber::_onEntityModify(BPEntityR el)
{
	return ERROR;
}
*/

bool ToolReNumber::_onModifierKeyTransition(bool wentDown, int key)
{
	return __super::_onModifierKeyTransition(wentDown, key);
}


BPTool* CreateToolReNumber()
{
	ToolReNumber* tool = new ToolReNumber();
	return tool;
	return NULL;
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("TestReNumber", &CreateToolReNumber);
AutoDoRegisterFunctionsEnd

