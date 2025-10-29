#include "stdafx.h"
#include "ToolCreateTableDemo.h"
#include "BPDisplay/BPTextProperties.h"
#include "BPDisplay/BPText.h"
#include "BPDisplay/BPFontUtil.h"
#include "BPPrimaryElement/BPTableEntity.h"
#include "BPPrimaryElement/BPTableEntityManager.h"
#include "BPPrimaryElement/BPMTextEntity.h"

using namespace BIMBase::Data;
using namespace BIMBase::Core;
using namespace DemoObject;

ToolCreateTableDemo::ToolCreateTableDemo()
{}

ToolCreateTableDemo::~ToolCreateTableDemo()
{}

void ToolCreateTableDemo::getCube(vector<DemoObject::CubeDemo>& vctCube)
{
	vctCube.clear();

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPEntityArray entityArray;
	BPEntityUtil::getEntities(entityArray, *pProject, PBM_SCHEMA_Demo, PBM_CLASS_CUBE_Demo);
	if (entityArray.getCount() == 0)
		return;

	for (int i = 0; i < entityArray.getCount(); i++)
	{
		BPEntityPtr curr = entityArray.getByIndex(i);
		if (!curr || !curr.isValid())
			continue;

		BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(*curr);
		if (!ptrData.isValid())
			continue;
		//根据根据实例初始化
		CubeDemo pbCube;
		pbCube.initFromData(*ptrData);
		vctCube.push_back(pbCube);
	}
}

void ToolCreateTableDemo::createTable()
{
	vector<DemoObject::CubeDemo> vctCube;
	getCube(vctCube);

	if (vctCube.size() == 0)
	{
		AfxMessageBox(_T("请先布置立方体"));
		return;
	}

	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;
	
	//创建表格
	BPTableEntityP table = new BPTableEntity;
	table->setRowNum(vctCube.size() + 2);
	table->setColumnNum(4);
	table->setColWidths(10000);
	table->setRowHeight(2000);
	table->addCellForTable();
	int hNum = table->m_colNum * (table->m_rowNum + 1);
	int vNum = table->m_rowNum * (table->m_colNum + 1);

	table->m_horizontalLineVisible.assign(hNum, true);

	table->m_verticalLineVisible.assign(vNum, true);
	
	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;
	BPTableEntityManager tablemanager = BPTableEntityManager(table);
	
	table->setTransform(p3d::GeTransform::create(0, 20000, 0));
	p3d::P3DStatus status = table->addToProject(*pProject, pModel->getModelId());

	auto fun = [&](PString sCur,bool bUnderlinedCur,bool bBold, UInt32 colorCur,
		double HangingIndentCur, int cellR,int cellC)
	{
		//创建一个文字
		BPMTextEntity pbmtext;
		BPFont font1 = BIMBase::Core::BPFontUtil::getDefaultTrueTypeFont();
		//多行文字之间的段落设置
		MTextParagraphPropertiesAppenderPtr ptrAppender = new MTextParagraphPropertiesAppender;
		ptrAppender->isFullJustified = false;
		ptrAppender->mtext_FirstLineIndent = 0;
		ptrAppender->mtext_HangingIndent = HangingIndentCur;
		//设置居中
		ptrAppender->justification = P3DTextEntityJustification::CenterMiddle;
		pbmtext.appendTextPart(ptrAppender);
		//多行文字属性设置
		MTextRunPropertiesAppenderPtr ptrRunAppender = new MTextRunPropertiesAppender;
		ptrRunAppender->color = colorCur;
		ptrRunAppender->isBold = bBold;
		ptrRunAppender->isItalic = false;
		ptrRunAppender->isUnderlined = bUnderlinedCur;
		ptrRunAppender->isOverlined = false;
		ptrRunAppender->font = font1;
		ptrRunAppender->overrideFontSize = true;
		ptrRunAppender->fontSize = GePoint2d::create(400, 600);
		pbmtext.appendTextPart(ptrRunAppender);
		//多行文字内容设置
		MTextTextLineAppenderPtr ptrTextAppender = new MTextTextLineAppender;
		ptrTextAppender->mtextLine = sCur;
		pbmtext.appendTextPart(ptrTextAppender);
		table->m_tableCells[cellR][cellC].m_context = pbmtext;
	};

	//表头
	fun(L"立方体属性表", false,true, 2,1, 0, 2);
	fun(L"序号", true, false, 0, 0,1, 0);
	fun(L"长度", false, false, 0, 0,1, 1);
	fun(L"宽度", true, false, 0,0, 1, 2);
	fun(L"高度", false, false, 0,0, 1, 3);

	BPTableCellEntity cell1, cell2;
	cell1.m_rowIndex = 0;
	cell1.m_colIndex = 0;
	cell2.m_rowIndex = 0;
	cell2.m_colIndex = 3;
	tablemanager.mergeCellsInTable(cell1, cell2);

	//表格内容
	for (int n = 0; n < vctCube.size(); n++)
	{
		CString str,strL,strW,strH;
		str.Format(_T("%d"),n+1);
		strL.Format(_T("%d"), vctCube[n].getLength());
		strW.Format(_T("%d"), vctCube[n].getWidth());
		strH.Format(_T("%d"), vctCube[n].getHeight());
		fun(str.GetString(), false, false, 0, 0, n + 2, 0);
		fun(strL.GetString(), false, false, 0, 0, n + 2, 1);
		fun(strW.GetString(), false, false, 0, 0, n + 2, 2);
		fun(strH.GetString(), false, false, 0, 0, n + 2, 3);
	}

	table->replaceInProject(*pProject);
}

static void registerTableDemo()
{
	ToolCreateTableDemo table;
	table.createTable();
}

AutoDoRegisterFunctionsBegin
BIMBase::BPToolsManager::registerFun("tableDemo", registerTableDemo);
AutoDoRegisterFunctionsEnd