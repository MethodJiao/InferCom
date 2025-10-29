#include "stdafx.h"
#include "ToolSelectBaseData.h"
#include "DataManagerTest.h"

using namespace TestObject;
using namespace ::p3d::platform;

ToolSelectBaseData::ToolSelectBaseData()
{

}


ToolSelectBaseData::~ToolSelectBaseData()
{

}

void ToolSelectBaseData::_onPostInstall()
{
	T_Super::_onPostInstall();

}

void   ToolSelectBaseData::_onRestartTool()
{
	ToolSelectBaseData* newTool = new ToolSelectBaseData();
	newTool->installTool();
}

void ToolSelectBaseData::_exitTool()
{

	__super::_exitTool();
}

bool ToolSelectBaseData::_onDataButton(BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);
	
	return true;

}

BIMBase::Core::BPEntityPtr ToolSelectBaseData::_buildLocateAgenda(BPPickDataCP path, BPBaseButtonEventCP ev)
{
	BPProjectP pProject = BPProject::getActiveProject();
	if (pProject == nullptr) {
		return nullptr;
	}
	BPEntityR ehd = BPEntity(path->GetHeadElem(), path->getModelBase());
	if (!ehd.isValid())
		return nullptr;
	GeTransform trans = GeTransform::createIdentityMatrix();
	BPGraphicsPtr ptrGra = BPEntityUtil::transformEntity(ehd, trans, false);
	for (BPGraphics::EntryPtr& load : *ptrGra)
	{
		BPGraphics::Entry::Type type = load->GetType();
		switch (load->GetType())
		{
		case  BPGraphics::Entry::Type::GeCurveBase:
		{
			IGeCurveBaseP pCu = load->getAsGeCurveBaseP();
			if (pCu == nullptr)
				return false;
			/*int size = cu->size();
			for (int j = 0; j < size; j++)
			{*/
				IGeCurveBase::CurveBaseType type = cu->getCurveBaseType();
				//拿出每一个面中线的信息
				if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_Segment)
				{
					//IGeCurveBasePtr curbase = pCu->at(j);
					//if (curbase != nullptr)
					//{
					//	GePoint3d pointA = GePoint3d::create(0, 0, 0);
					//	GePoint3d pointB = GePoint3d::create(0, 0, 0);
					//	curbase->getStartEndPoint(pointA, pointB);
					//	//AfxMessageBox(L"起点，重点");
					//}

				}
				else if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_LineString)
				{
					IGeCurveBasePtr ptrCurbase = pCu;
					if (ptrCurbase != nullptr)
					{
						const pvector<GePoint3d>* pts = ptrCurbase->getLineStringCP();
						GePoint3d pointA = pts->at(0);

						GePoint3d pointB = pts->at(pts->size() - 1);
						double length = pointA.distance(pointB);
						CString str = _T("");
						str.Format(_T("长度是%f"), length);
						AfxMessageBox(/*(LPCTSTR)*/str);

					}
				}
			//}
			break;
		}
		case BPGraphics::Entry::Type::GeSolidBase:
		{
			IGeSolidBaseP pSolid = load->getAsGeSolidBaseP();
			if (pSolid == nullptr)
				return false;
			//拿到几何体的点，线，面
			pvector <GeSolidLocationInfo::GeFaceIndices> indices;
			//得到soild里面indices
			pSolid->getFaceIndices(indices);
			pvector<GeCurveArrayPtr> curearray;
			//通过indices拿到对应的面
			for (auto indice : indices)
			{
				IGeometryPtr ptrGeom = pSolid->getFace(indice);
				if (ptrGeom.isValid())
				{
					GeCurveArrayPtr ptrCv = ptrGeom->getAsGeCurveArray();
					if (ptrCv.isValid())
					{
						curearray.push_back(ptrCv);
					}
				}
			}
			GeCurveArrayPtr ptrCu = curearray[0];//拿出几何体第一个面：底面
			int size = ptrCu->size();
			for (int j = 0; j < size; j++)
			{
				IGeCurveBase::CurveBaseType type = ptrCu->at(j)->getCurveBaseType();
				//拿出每一个面中线的信息
				if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_Segment)
				{
					IGeCurveBasePtr ptrCurbase = ptrCu->at(j);
					if (ptrCurbase != nullptr)
					{
						GePoint3d pointA = GePoint3d::create(0, 0, 0);
						GePoint3d pointB = GePoint3d::create(0, 0, 0);
						ptrCurbase->getStartEndPoint(pointA, pointB);
						//AfxMessageBox(L"起点，重点");
					}

				}
				else if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_LineString)
				{
					IGeCurveBasePtr ptrCurbase = ptrCu->at(j);
					if (ptrCurbase != nullptr)
					{
						const pvector<GePoint3d>* pts = ptrCurbase->getLineStringCP();
						GePoint3d pointA = pts->at(0);
						GePoint3d pointC = pts->at(1);
						GePoint3d pointB = pts->at(2);
						double length = pointA.distance(pointC);
						double width = pointC.distance(pointB);
						CString str = _T("");
						str.Format(_T("长度是%f,宽度是%f"), length,width);
						AfxMessageBox(/*(LPCTSTR)*/str);

					}
				}
			}
			break;
		}

		}

	}
	return nullptr;

}
bool ToolSelectBaseData::_onResetButton(BPBaseButtonEventCP ev)
{
	// 获取当前工程
	_exitTool();
	return true;
}

p3d::StatusInt ToolSelectBaseData::_onEntityModify(BPEntityR el)
{
	return ERROR;
}


BPTool* CreateToolSelectBaseData()
{
	ToolSelectBaseData* tool = new ToolSelectBaseData();
	return tool;
	
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("ToolSelectBaseData", &CreateToolSelectBaseData);
AutoDoRegisterFunctionsEnd

