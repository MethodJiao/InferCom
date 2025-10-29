#include "stdafx.h"
#include "ToolBaseDataSaveDemo.h"
#include "DataManagerDemo.h"
#include "TgGe/gepnt3d.h"
#include "TgGe/acgetotgge.h"
#include "PBBimCore/PBTgGe.h"
#include "LineDemo.h"
using namespace DemoObject;
using namespace ::p3d::platform;

ToolBaseDataSaveDemo::ToolBaseDataSaveDemo()
{
	
}


ToolBaseDataSaveDemo::~ToolBaseDataSaveDemo()
{
	
}

void ToolBaseDataSaveDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"点击对象"));
}

void   ToolBaseDataSaveDemo::_onRestartTool()
{
	ToolBaseDataSaveDemo* newTool = new ToolBaseDataSaveDemo();
	newTool->installTool();
}



bool ToolBaseDataSaveDemo::_onDataButton(BPBaseButtonEventCP ev)
{
	//获取鼠标屏幕点击的点
	GePoint3d ptCur = *ev->getPoint();
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == nullptr)
		return false;
	//获取点击点所在的工程和模型ID
	::BIMBase::PModelId curModelId = ev->getViewport()->getTargetModel()->getModelId();
	//管理线，立方体的数据并存起来
	m_BaseData.clear();
	DemoObject::LineDemoP pLine =  new DemoObject::LineDemo();
	if (pLine == nullptr)
		return false;
	DemoObject::SoildCubeDemoP Soild = new SoildCubeDemo();
	if (Soild == nullptr)
		return false;
	pvector<DemoObject::BaseDataDemoP> BaseData;
	BaseData.push_back(pLine);
	//BaseData.push_back(Soild);
	
	for (size_t i = 0; i < BaseData.size(); i++)
	{

		BaseDataDemoP pTe = BaseData.at(i);
		DataManagerDemo::Get().addPhysicalGraphics(*pProject, BaseData.at(i));
		if (SUCCESS != DataManagerDemo::Get().addToProject(*pProject, curModelId))
		{
			AfxMessageBox(L"Can not add to project!");
		}
	}
	AfxMessageBox(L"已储存");
	return true;
	
}


bool ToolBaseDataSaveDemo::_onResetButton(BPBaseButtonEventCP ev)
{
	_exitTool();
	return true;
}

BPTool* CreateToolBaseDataSaveDemo()
{
	ToolBaseDataSaveDemo* tool = new ToolBaseDataSaveDemo();
	return tool;
	
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("baseDataSaveDemo", &CreateToolBaseDataSaveDemo);
AutoDoRegisterFunctionsEnd



ToolBaseDataGet::ToolBaseDataGet()
{

}


ToolBaseDataGet::~ToolBaseDataGet()
{

}

void ToolBaseDataGet::_onPostInstall()
{
	T_Super::_onPostInstall();
	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"点击对象"));
}

void   ToolBaseDataGet::_onRestartTool()
{
	ToolBaseDataGet* newTool = new ToolBaseDataGet();
	newTool->installTool();
}


//取数据
bool ToolBaseDataGet::_onDataButton(BPBaseButtonEventCP ev)
{
	BPProjectP pProject = ev->getViewport()->getTargetModel()->getBPProject();
	if (pProject == NULL)
		return false;
	pvector<byte*> bys = DataManagerDemo::Get().getAllByte();
	size_t si = bys.size();
	//记录当前有几个line，soild
	int nCout_line = 0;
	int nCout_soild = 0;
	pvector<int> strsize = DataManagerDemo::Get().getStringVec();
	pvector<int> bysize = DataManagerDemo::Get().getByteVec();
	CString  sName = DataManagerDemo::Get().getClassName();
	PModelId modelId = pProject->getDefaultModelId();
	BIMBase::Core::BPModelP pModel = pProject->loadModelById(modelId);
	if (pModel == NULL)
		return false;
	BPGraphicsPtr ptrTempGraphics = pModel->createPhysicalGraphics();
	GeTransform trans = DataManagerDemo::Get().getPlacement().toTransform();
	BPGraphicsUtils::transformPhysicalGraphics(*ptrTempGraphics, trans);
	if (!ptrTempGraphics.isValid())
		return false;

	for (size_t j = 0; j < si; j++)
	{
		byte* by = bys.at(j);
		int stringsize = strsize.at(j);
		int bytesize = bysize.at(j);
#if 1		//cereal方法
		if (sName == L"LineDemo")
		{
			vector<char> binaryDataVec;

			for (int i = 0; i < bytesize; i++)
			{

				binaryDataVec.push_back((char)by[i]);
			}

			if (!DataManagerDemo::Get().deSerialize_Cereal(binaryDataVec))
				return false;
			std::vector<GePoint3d> pts = DataManagerDemo::Get().getPts();

			int size = pts.size();
			if (size > 0)
			{
				GeSegment3d seg = GeSegment3d::create(pts[0], pts[size - 1]);
				IGeCurveBasePtr curve = IGeCurveBase::createSegment(seg);
				ptrTempGraphics->addGeCurve(*curve);
				ptrTempGraphics->finish();
				LineDemoP line = new LineDemo();
				line->m_ptrLineGraphics = ptrTempGraphics;
				BPGraphicsPtr gra = line->createGraphics();

				nCout_line++;
				for (BPGraphics::EntryPtr& load : *gra)
				{
					BPGraphics::Entry::Type type = load->getType();
					switch (load->getType())
					{
					case  BPGraphics::Entry::Type::GeCurveBase:
					{
						IGeCurveBaseP pCu = load->getAsGeCurveBaseP();
						if (pCu == nullptr)
							return false;

						IGeCurveBase::CurveBaseType type = pCu->getCurveBaseType();
						//拿出每一个面中线的信息
						if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_Segment)
						{
							IGeCurveBasePtr ptrCurbase = pCu;
							if (ptrCurbase != nullptr)
							{
								GePoint3d pointA = GePoint3d::create(0, 0, 0);
								GePoint3d pointB = GePoint3d::create(0, 0, 0);
								ptrCurbase->getStartEndPoint(pointA, pointB);
								double length = pointA.distance(pointB);
								CString str = _T("");
								str.Format(_T("线段长度是%f"), length);
								AfxMessageBox(str);
							}

						}
						else if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_LineString)
						{
							IGeCurveBasePtr curbase = pCu;
							if (curbase != nullptr)
							{
								const pvector<GePoint3d>* pts = curbase->getLineStringCP();
								GePoint3d pointA = pts->at(0);

								GePoint3d pointB = pts->at(pts->size() - 1);
								double dlength = pointA.distance(pointB);
								CString str = _T("");
								str.Format(_T("长度是%f"), dlength);
								AfxMessageBox(/*(LPCTSTR)*/str);

							}
						}
						break;
					}
					}
				}
			}
		}

#else	//CArchive方法
		CMemFile memFile;
		memFile.SeekToBegin();
		CArchive ar(&memFile, CArchive::store);
		CString name;

		for (int i = stringsize; i < bytesize; i++)
		{
			ar << by[i];
		}
		ar.Close();
		memFile.SeekToBegin();
		CArchive arLoad(&memFile, CArchive::load);

		PBTgGe::SerializePhysicalElement(tempGraphics, arLoad);

		tempGraphics->finish();

		CMemFile memFile1;
		memFile1.SeekToBegin();
		CArchive ar1(&memFile1, CArchive::store);


		for (int i = 0; i < stringsize; i++)
		{
			ar1 << by[i];
		}
		ar1.Close();
		memFile1.SeekToBegin();
		CArchive arLoad1(&memFile1, CArchive::load);
		arLoad1 >> name;
		if (name == L"LineDemo")
		{
			LineDemoP line = new LineDemo();
			line->m_lineGraphics = tempGraphics;
			BPGraphicsPtr gra = line->createGraphics();

			cout_line++;
			for (BPGraphics::EntryPtr& load : *gra)
			{
				BPGraphics::Entry::Type type = load->getType();
				switch (load->getType())
				{
				case  BPGraphics::Entry::Type::GeCurveBase:
				{
					IGeCurveBaseP cu = load->getAsGeCurveBaseP();
					if (cu == nullptr)
						return false;

					IGeCurveBase::CurveBaseType type = cu->getCurveBaseType();
					//拿出每一个面中线的信息
					if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_Segment)
					{
						IGeCurveBasePtr curbase = cu;
						if (curbase != nullptr)
						{
							GePoint3d pointA = GePoint3d::create(0, 0, 0);
							GePoint3d pointB = GePoint3d::create(0, 0, 0);
							curbase->getStartEndPoint(pointA, pointB);
							double length = pointA.distance(pointB);
							CString str = _T("");
							str.Format(_T("线段长度是%f"), length);
							AfxMessageBox(str);
						}

					}
					else if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_LineString)
					{
						IGeCurveBasePtr curbase = cu;
						if (curbase != nullptr)
						{
							const pvector<GePoint3d>* pts = curbase->getLineStringCP();
							GePoint3d pointA = pts->at(0);

							GePoint3d pointB = pts->at(pts->size() - 1);
							double length = pointA.distance(pointB);
							CString str = _T("");
							str.Format(_T("长度是%f"), length);
							AfxMessageBox(/*(LPCTSTR)*/str);

						}
					}
					break;
				}
				}
			}
		}
		else if (name == L"SoildCubeDemo")
		{
			SoildCubeDemoP soild = new SoildCubeDemo();
			soild->m_soildGraphics = tempGraphics;
			BPGraphicsPtr gra = soild->createGraphics();
			gra->save();
			cout_soild++;
		}


#endif
	}
	CString str = _T("");
	str.Format(_T("当前有%d个Line"), nCout_line);
	AfxMessageBox(str);
	return true;

}


bool ToolBaseDataGet::_onResetButton(BPBaseButtonEventCP ev)
{
	_exitTool();
	return true;
}

BPTool* CreateToolBaseDataGet()
{
	ToolBaseDataGet* tool = new ToolBaseDataGet();
	return tool;

}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("baseDataGetDemo", &CreateToolBaseDataGet);
AutoDoRegisterFunctionsEnd