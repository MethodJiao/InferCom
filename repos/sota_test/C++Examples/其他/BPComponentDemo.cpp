#include "stdafx.h"

using namespace BIMBase::ParaComponent;

//导入bfa，进行布置
void BPComponentDemo()
{
	string bfaPath = "D:\\房子.bfa";
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelPtr ptrModel = BPModel::getActiveModel();
	if (!ptrModel)
		return;
	BIMBase::ParaComponent::BPComponentBfaHandle bfaHandle = BIMBase::ParaComponent::BPComponentBfaManager::addImported(pProject, bfaPath);
	if (!bfaHandle.valid())
		return;
	
	std::set<BPComponentBfaHandle> hanset = BPComponentBfaManager::getImportedAll(pProject);
	if (hanset.empty())
		return;
	for (auto item : hanset)
	{
		if (!item.valid())
			continue;
		string st = item.getCategoryName();

	}
	string savePath = "D:\\房子2.bfa";
	if (BPComponentBfaManager::saveBfaToFile(bfaHandle, savePath))
	{
		AfxMessageBox(L"保存成功");
	}
	std::vector<std::string> listCategoryNameList = BPComponentBfaManager::getCategoryNameList();
	for (auto item : listCategoryNameList)
	{
		string schemaName = "";
		string className = "";
		if (!BPComponentBfaManager::getSchemaByCategoryName(item, schemaName, className))
		{
			AfxMessageBox(L"获取类别名称对应schema和class失败");
		}
	}
	 
	BPComponentBfaEditHandle editHandle = BPComponentBfaManager::createBfaEditingObject(bfaHandle);
	if (!editHandle.valid())
		return;
	bool isEditingEnvironment = BPComponentBfaManager::isEditingEnvironment();
	string savePath1 = "D:\\房子3.bfa";
	if (BPComponentBfaManager::saveBfaToFile(editHandle, savePath1))
	{
		AfxMessageBox(L"保存成功");
	}
	std::vector<std::string> listTypeName = bfaHandle.getTypeNameList();
	if (listTypeName.size() == 0)
		return;
	BPComponentBfacomponentEditer editer = bfaHandle.createBfacomponentEditer(listTypeName.at(0));
	BPComponentBfaPlacedEditHandle placeEditHandle1 = BPComponentBfaManager::placeObject(pProject, editer.getObject(), ptrModel.get());
	if (!placeEditHandle1.valid())
		return;
	std::set<BPComponentBfaPlacedEditHandle> listPlacedEditHandle = BPComponentBfaManager::getPlacedHandle(pProject);
	std::set<BPComponentBfaPlacedEditHandle> listPlacedEditHandleByType = BPComponentBfaManager::getPlacedHandleFromType(pProject, bfaHandle, listTypeName.at(0));
	
	BPComponentBfacomponentEditer impediter = bfaHandle.createBfacomponentEditer("房子");
	BPComponentBfaPlacedEditHandle implaceEditHandle = BPComponentBfaManager::placeObject(pProject, impediter.getObject(), ptrModel.get());
	if (!implaceEditHandle.valid())
		return;
	int count = 0;
	for (auto item : listPlacedEditHandle)
	{
		if (BPComponentBfaManager::delPlaced(pProject, item))
		{
			count++;
		}
	}
	CString str = _T("");
	str.Format(_T("共删除%d个实例成功"), count);
	AfxMessageBox(str);
}
//增加设置属性
void  testComponentProperty()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	BPModelPtr ptrModel = BPModel::getActiveModel();
	if (!ptrModel)
		return;
	std::set<BPComponentBfaPlacedEditHandle> edithandset = BPComponentBfaManager::getPlacedHandle(pProject);
	if (edithandset.empty())
		return;
	BPComponentBfaPlacedEditHandle placehand = *edithandset.begin();
	if (!placehand.valid())
		return;
	BIMBase::ParaComponent::Gnrc g = placehand.getPropertyValue(0);
    std:map<BPPropertyID, BIMBase::ParaComponent::Gnrc >val;
	val.insert(std::make_pair(1, string("新属性名称")));
	val.insert(std::make_pair(2, long long(999)));
	placehand.setPropertyValue(val);
	BIMBase::ParaComponent::Gnrc g1 = placehand.getPropertyValue(1);
	std::set<BPComponentBfaHandle> importhandleset = BPComponentBfaManager::getImportedAll(pProject);
	if (importhandleset.empty())
		return;
	BPComponentBfaHandle importhand = *importhandleset.begin();
	BIMBase::ParaComponent::Gnrc g2 = importhand.getPropertyValue("默认类型1", 1);

	BPComponentBfaEditHandle iseditHandle = BPComponentBfaManager::getCurrentlyEditingHandle();
	if (!iseditHandle.valid())
		return;
	string name = iseditHandle.getName();
	iseditHandle.setName(name + "-test");
	string categoryName = iseditHandle.getCategoryName();
	iseditHandle.setCategoryName(categoryName + "-test");
	if (iseditHandle.addType("test类型1") && iseditHandle.addType("test类型2"))
	{
		AfxMessageBox(L"新建类型成功");
	}
	std::vector<std::string> listTypeName = iseditHandle.getTypeNameList();
	if (iseditHandle.delType("test类型1"))
	{
		AfxMessageBox(L"删除类型成功");
	}
	if (iseditHandle.setTypeName("test类型2", "test类型11"))
	{
		AfxMessageBox(L"修改类型成功");
	}
	std::vector<BPPropertyID> listPropertyId = iseditHandle.getPropertyIdAll();
	for (BPPropertyID item : listPropertyId)
	{
		bool isType = iseditHandle.isTypeProperty(item);
		bool isInstance = iseditHandle.isInstanceProperty(item);
		BPComponentBfaPropertyType type = iseditHandle.isPropertyType(item);
		string propertyKey = iseditHandle.getPropertyKey(item);
		BPPropertyID id = iseditHandle.getPropertyId(propertyKey);
		string propertyGroup = iseditHandle.getPropertyGroup(item);
		string des = iseditHandle.getPropertyDescription(item);
	}
	BPPropertyID newId = iseditHandle.addProperty("字符串属性");
	iseditHandle.setPropertyKey(newId, "字符串属性名称");
	iseditHandle.setPropertyGroup(newId, "字符");
	iseditHandle.setPropertyDescription(newId, "这是字符串");
	newId = iseditHandle.addProperty(false);
	iseditHandle.setPropertyKey(newId, "布尔属性名称");
	newId = iseditHandle.addProperty(520.1314);
	iseditHandle.setPropertyKey(newId, "小数属性名称");
	newId = iseditHandle.addProperty(long long(888666));
	iseditHandle.setPropertyKey(newId, "整数名称");
}

//测试内置类别添加
void testComponentInnerCategory()
{
	BPComponentBfaManager::enrolCategory("ParaCom","PBM_CoreModel","BPParaComponentDefault",false);
	std::vector<std::tuple<BIMBase::ParaComponent::Gnrc, std::map<BPParaLanguage, std::string>>> Property;
	std::tuple<BIMBase::ParaComponent::Gnrc, std::map<BPParaLanguage, std::string>> tepara;
	std::map<BPParaLanguage, std::string> mapopara;
	mapopara[BPParaLanguage::Chinese] = "内置ParaCom属性1";
	tepara = { long long(1000),mapopara };
	Property.push_back(tepara);
	std::vector <innerPropertyInformation> PropertyInfo1;
	innerPropertyInformation info2;
	info2.m_value = string("内置ParaCom属性1");
	info2.m_isReadOnly = true;
	info2.m_isShow = true;
	info2.m_isTypePrperty = false;
	std::map<BPParaLanguage, std::string> map1;
	map1 [BPParaLanguage::Chinese]= "内置ParaCom属性1stringChinese";
	map1[BPParaLanguage::English] = "内置ParaCom属性1stringEnglish";
	info2.m_languageAndPropertyName = map1;
	PropertyInfo1.push_back(info2);
	BPComponentBfaManager::enrolInnerPropertyByCategory("ParaCom", PropertyInfo1);

}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("ComponentDemo", &BPComponentDemo);
BPToolsManager::registerFun("ComponentProperty", &testComponentProperty);
BPToolsManager::registerFun("ComponentInnerCategory", &testComponentInnerCategory);
AutoDoRegisterFunctionsEnd