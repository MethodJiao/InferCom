#include "stdafx.h"
#include "MyCubeDragManipulatorDemo.h"
#include "CubeDemo.h"
#include "adsarxfunc\ADSARXFUN.h"

#define msGeomConst_piOver2     1.57079632679489660000e+000

using namespace DemoObject;


DragRectTriangle::DragRectTriangle(GeVec3dCR direction, GeVec3dCR rotate, GePoint3dCR point, int length, int menuColor, int transparency)
{
	m_point = point;
	m_state = BPPointDragManipulator::ControlState::CONTROL_STATE_UnSelected;
	m_direction = direction;
	m_rotate = rotate;
	m_nLength = length;
	m_menuColor = menuColor;
	m_transparency = transparency;
}

//绘制三角夹点
void DragRectTriangle::_draw(BPViewportR vp, GeTransformCP transform) const
{
	GePoint3d ptPoint = m_point;
	GeVec3d vecDir = m_direction;
	GeVec3d vecVer = m_rotate;
	vecDir.normalize();
	vecVer.normalize();

	if (transform)
	{
		transform->multiply(ptPoint);
		transform->matrixOnlyMultiply(vecDir);
		transform->matrixOnlyMultiply(vecVer);
	}
	uint32_t lineColor = vp.makeTrgbColor(127, 191, 255, 0);
	uint32_t fillColor = vp.makeTrgbColor(127, 191, 255, 5);
	int color = (BPPointDragManipulator::ControlState::CONTROL_STATE_UnSelected == m_state ? fillColor : vp.getHiliteColor());

	int nWidth = vp.getPixelSizeAtPoint(&ptPoint, P3DCoordSystem::enWorld) * m_nLength;
	GePoint3d ptsTriangle[3];
	ptPoint = ptPoint + vecVer * 20;
	ptsTriangle[0] = (ptPoint + vecVer * nWidth / 2);
	ptsTriangle[1] = (ptPoint + vecVer * -nWidth / 2);
	ptsTriangle[2] = ptPoint + vecDir * nWidth;

	vp.getIViewOutput()->setSymbology(lineColor, color, 2, 0);
	vp.getIViewOutput()->drawShape3d(3, ptsTriangle, true, NULL);
}

bool DragRectTriangle::_locate(::BIMBase::Core::BPBaseButtonEventCR ev, p3d::GeTransformCP transform) const
{
	BPViewportP pVp = ev.getViewport();
	if (!pVp)
		return false;
	GePoint3d ptPoint = m_point;
	GeVec3d vecDir = m_direction;
	GeVec3d vecVer = m_rotate;
	vecDir.normalize();
	vecVer.normalize();
	double scaWidth = 1;
	if (transform)
	{
		transform->multiply(ptPoint);
		transform->matrixOnlyMultiply(vecDir);
		transform->matrixOnlyMultiply(vecVer);
		GePoint3d fixedPoint = GePoint3d::createByZero();
		transform->isUniformScale(fixedPoint, scaWidth);
	}
	pvector<GePoint3d> vecPoints;
	GePoint3d ptScale = GePoint3d::createByZero();
	P3DStatus pOut = pVp->getScale(ptScale);

	int nWidth = m_nLength / ptScale.x / scaWidth;
	GePoint3d ptsTriangle[3];
	ptPoint = ptPoint + vecVer * 20;
	ptsTriangle[0] = (ptPoint + vecVer * nWidth / 2);
	ptsTriangle[1] = (ptPoint + vecVer * -nWidth / 2);
	ptsTriangle[2] = ptPoint + vecDir * nWidth;

	GeTransform tran;
	tran.setByOriginAndVectors(m_point, m_direction, m_rotate);
	tran.setByInverse(tran);
	GePoint3d pt1 = GePoint3d::createByZero();
	GePoint3d pt2 = GePoint3d::createByZero();
	GePoint3d pt3 = GePoint3d::createByZero();
	pt1 = ptsTriangle[0];
	pt2 = ptsTriangle[1];
	pt3 = ptsTriangle[2];
	tran.multiply(pt1);
	tran.multiply(pt2);
	tran.multiply(pt3);
	vecPoints.push_back(pt1);
	vecPoints.push_back(pt2);
	vecPoints.push_back(pt3);
	vecPoints.push_back(pt1);
	IGeCurveBasePtr ptrCuprimitive = IGeCurveBase::createLineString(vecPoints);
	GeCurveArrayPtr ptrCurve = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	ptrCurve->add(ptrCuprimitive);
	GePoint3d pt = *ev.getPoint();
	GeRay3d rayEye;
	InitBoresite(rayEye, pt, pVp->getRootToViewMap()->M1);
	GePlane3d plane = GePlane3d::createByThreePoints(ptsTriangle[0], ptsTriangle[1], ptsTriangle[2]);
	double dpara = 0;
	GePoint3d pttemp = GePoint3d::createByZero();
	if (rayEye.getIntersectPointWithPlane(pttemp, dpara, plane))
	{
		pt = pttemp;
	}
	tran.multiply(pt);
	GeCurveArray::InOutClassification pos = ptrCurve->getPointLocationType2D(pt);
	if (GeCurveArray::INOUT_In == pos || GeCurveArray::INOUT_On == pos)
	{
		return true;
	}
	return false;

}
static bool trig_safeDiveide(double* pRatio, double numerator, double denominator, double defaultRatio)
{
	static double s_relTOl = 1.0e-12;
	if (fabs(denominator) > s_relTOl * fabs(numerator))
	{
		*pRatio = numerator / denominator;
		return true;
	}
	else
	{
		*pRatio = defaultRatio;
		return false;
	}
}
void DragRectTriangle::InitBoresite(GeRay3dR boresite, GePoint3dCR spacePoint, GeMatrix4dCR worldToLocal)
{
	boresite.origin = spacePoint;
	GePoint4d eyePoint;
	worldToLocal.getColumn(eyePoint, 2);
	boresite.direction.set(eyePoint.x, eyePoint.y, eyePoint.z);
	double aa;
	if (trig_safeDiveide(&aa, 1.0, eyePoint.w, 1.0))
	{
		GePoint3d xyzEye = GePoint3d::createByZero();
		xyzEye = boresite.direction * aa;
		boresite.direction = xyzEye - boresite.origin;
	}
	boresite.direction.normalize();
	boresite.direction.negate();
}



MyCubeDragManipulatorDemo::MyCubeDragManipulatorDemo()
{
}


MyCubeDragManipulatorDemo::~MyCubeDragManipulatorDemo()
{
}

int MyCubeDragManipulatorDemo::addDragControlArrow(GePoint3dCR point, GeVec3dCR vecDirection, GeVec3dCR vecSide)
{
	m_controls.m_locations.push_back(new DragRectTriangle(vecDirection, vecSide, point));

	return (m_controls.m_locations.size() - 1);
}

int MyCubeDragManipulatorDemo::addControlArrowPoint(GePoint3dCR point, GeVec3dCR vecDir)
{
	m_controls.m_locations.push_back(new ControlArrow(point, vecDir));
	return (m_controls.m_locations.size() - 1);
}

bool MyCubeDragManipulatorDemo::_onCreateControls(::BIMBase::Core::BPEntityCR eh)
{
	//根据传入的BPEntity信息获取对象实例
	BIMBase::Core::BPDataPtr ptrData = BPDataUtil::getDataOnEntity(eh);
	if (!ptrData.isValid())
		return false;

	//根据根据实例初始化
	CubeDemo pbCube;
	pbCube.initFromData(*ptrData);
	
	GePoint3d startPt, endPt, midPt;
	startPt = pbCube.getStartPoint();
	endPt = pbCube.getEndPoint();
	midPt = pbCube.getMiddlePoint();

	// 基础数据
	double dHeight = pbCube.getHeight();

	//原默认夹点
	addControlPoint(startPt);
	addControlPoint(endPt);
	addControlPoint(midPt);
	
	GeVec3dCR vecDir = GeVec3d::create(0, 1000, 0);
	GeVec3dCR vecSide = GeVec3d::create(0, 0, 0);

	return true;
}

StatusInt MyCubeDragManipulatorDemo::_doDragControls(BPEntityR elHandle, BPBaseButtonEventCR ev, bool isDynamics)
{
	//根据传入的BPEntity信息获取对象实例
	BPDataPtr ptrData = BPDataUtil::getDataOnEntity(elHandle);
	if (!ptrData.isValid())
		return false;

	//根据实例初始化
	CubeDemo pbCube;
	pbCube.initFromData(*ptrData);

	//获取点击夹点后鼠标点
	GePoint3d cursorPt = GePoint3d::create(0, 0, 0);
	_getAdjustedPoint(cursorPt, ev);
	

	//获取点击夹点的编号，编号根据加入夹点的顺序确定
	GePoint3d m_orginPt;
	SIZE_T index = -1;
	for (SIZE_T i = 0; i < m_controls.m_locations.size(); i++)
	{
		if (CONTROL_STATE_Flashed == m_controls.m_locations[i]->m_state)
		{
			m_orginPt = m_controls.m_locations[i]->m_point;
			index = i;
			break;
		}
	}
	if (index < 0)
		return ERROR;

	cursorPt.z = m_orginPt.z;
	//通过转换矩阵获取原点及坐标信息
	GeTransform curTm = pbCube.getPlacement().toTransform();
	GePoint3d ptOri;
	GeVec3d vecX, vecY, vecZ;
	curTm.getOriginAndVectors(ptOri, vecX, vecY, vecZ);

	double dLenNew = pbCube.getLength();
	GeVec3d vecXNew = vecX;
	GePoint3d ptOriNew = ptOri;
	CubeDemo newbCube;
	newbCube.initFromData(*ptrData);
	GePoint3d ptEnd = pbCube.getEndPoint();

	if (index == 0)
	{
		// 起点
			//根据新起点重新计算墙长度及坐标系
		dLenNew = cursorPt.distance(ptEnd);
		vecXNew = GeVec3d::createByStartEndNormalize(cursorPt, ptEnd);
		ptOriNew = cursorPt;
		GeVec3d vecYNew = vecZ ^ vecXNew;
		//根据新计算的坐标及原点得到转换矩阵
		GeTransform transNew = GeTransform::createByOriginAndVectors(ptOriNew, vecXNew, vecYNew, vecZ);

		BPPlacement placNew;
		placNew.fromTransform(transNew);
		//墙设置新的转换矩阵，转换到新的位置
		pbCube.setPlacement(placNew);
		pbCube.setLength(dLenNew);
	}
	else if (index == 1)//end点，设置为点击的时候再布置一个立方体，立方体的长度随着鼠标移动而改变
	{
		
		dLenNew = cursorPt.distance(ptEnd);
		vecXNew = GeVec3d::createByStartEndNormalize(ptEnd, cursorPt);
		GeVec3d vecYNew = vecZ ^ vecXNew;
		//根据新计算的坐标及原点得到转换矩阵
		GeTransform transNew = GeTransform::createByOriginAndVectors(ptEnd, vecXNew, vecYNew, vecZ);
		BPPlacement placNew;
		placNew.fromTransform(transNew);
		//placNew.setOrigin(ptEnd);
		newbCube.setPlacement(placNew);
		newbCube.setLength(dLenNew);

	}
	else
	{
		//根据鼠标移动点与原始点求转换矩阵
		GePoint3d offset = { cursorPt.x - m_orginPt.x, cursorPt.y - m_orginPt.y, cursorPt.z - m_orginPt.z };
		GeTransform tm = GeTransform::create(offset);
		pbCube.onTransform(tm);
	}

	::BIMBase::PModelId curModelId = ev.getViewport()->getTargetModel()->getModelId();

	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return ERROR;
	BPProjectP pProject = pProjectManager->getMainProject();
	if (pProject == NULL)
		return ERROR;

	if (isDynamics)
	{//动态显示
		if (index != 1)
		{
			BPGraphicsPtr  ptrGraphics = pbCube.createPhysicalGraphics(*pProject, curModelId, true);
			if (ptrGraphics.isValid())
			{
				BIMBase::Data::BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, pbCube.getPlacement().toTransform());
				elHandle = ptrGraphics->getEntityR();
			}
		}
		else
		{
			BPGraphicsPtr  ptrGraphics = newbCube.createPhysicalGraphics(*pProject, curModelId, true);
			if (ptrGraphics.isValid())
			{
				BIMBase::Data::BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphics, newbCube.getPlacement().toTransform());
				elHandle = ptrGraphics->getEntityR();
			}

		}
		
	}
	else
	{//最终点击后Replace
		if (index != 1)
			return pbCube.replaceInProject(*pProject);
		else
			return newbCube.addToProject(*pProject, curModelId);
	}

	return SUCCESS;
}


BPGraphicsPtr MyCubeDragManipulatorDemo::createDimensionLinear(CubeDemoP pbCube, BPProjectR project, BPViewportP viewport)
{
	if (pbCube == nullptr)
		return nullptr;

	GePoint3d startPt, endPt;
	startPt = pbCube->getStartPoint();
	endPt = pbCube->getEndPoint();
	
	//标注样式
	BIMBase::Core::BPDimensionStylePtr dimensionStyle = BIMBase::Core::BPDimensionStyle::create(L"标注样式", project);
	if (dimensionStyle.isNull())
		return nullptr;

	dimensionStyle->setDimtad(1);
	dimensionStyle->setDimse1(true);
	dimensionStyle->setDimse2(true);
	dimensionStyle->setDimdec(0);
	dimensionStyle->setDimrnd(0);
	dimensionStyle->setDimscale(1);
	dimensionStyle->setDimtxt(1000);

	BIMBase::BPColorDef colorDef;
	colorDef.m_rgba.red = 125;
	colorDef.m_rgba.green = 125;
	colorDef.m_rgba.blue = 125;

	UInt32 colorInt = BPColorUtil::getEntityColor(colorDef, project, true);
	dimensionStyle->setDimclrd(colorInt);
	dimensionStyle->replace(L"标注样式", &project);

	//绘制选中对象的标注
	BIMBase::Data::BPPlacement dimPlace;
	GePoint3d xLine1Point = startPt;
	GePoint3d xLine2Point = endPt;
	GeVec3d dirVec = GeVec3d::createByStartEndNormalize(xLine1Point, xLine2Point);
	double dAng = GeVec3d::create(1., 0., 0.).angleTo2D(dirVec);
	GeVec3d vOffset = dirVec;
	vOffset.rotate2D(msGeomConst_piOver2);

	p3d::GeRotMatrix rm = p3d::GeRotMatrix::createIdentityMatrix();
	viewport->getRotation(rm);
	GeVec3d viewDir, zAxis;
	rm.getRow(viewDir, 2);
	zAxis.create(0, 0, 1);
	double dp1 = viewDir * vOffset;
	double dp2 = viewDir * zAxis;
	if (dp1 * dp2 < 0)
		vOffset.negate();

	vOffset = vOffset * 2000;
	GePoint3d midPoint = xLine1Point;
	dirVec = dirVec * 0.5;
	midPoint = midPoint + dirVec;
	midPoint = midPoint + vOffset;

	//根据计算信息构造直线标注
	BIMBase::Data::BPDimensionLinear linearDim(dimPlace, midPoint, xLine1Point, xLine2Point, midPoint, L"", dAng);
	linearDim.setDimstyle(L"标注样式");

	linearDim.addToModel(project, viewport->getTargetModel()->getModelId());

	BPGraphicsPtr  ptrGraphics = linearDim.createPhysicalGraphics(project, viewport->getTargetModel()->getModelId(), true);
	//graphicsPtr->save();
	return ptrGraphics;
}


void MyCubeDragManipulatorDemo::_onDraw(BPEntityCR element, BPViewportP viewport)
{
	T_Super::_onDraw(element, viewport);

	//获取选中的对象
	BPProjectR project = *viewport->getTargetModel()->getBPProject();
	BIMBase::Data::IBPObjectPtr obj = BPObjectExtensionManager::getInstance().getBPObject(element);

	if (!obj.isValid())
		return;

	//对象动态转换为墙
	CubeDemoP pbCube = dynamic_cast<CubeDemoP>(obj.get());
	if (!pbCube)
		return;

	BPGraphicsPtr  ptrGraphics;

	if (!ptrGraphics.isValid())
		return;

	//临时绘制
	BPViewDrawP viewDraw = viewport->getIViewDraw();
	uint32_t dimLineColor = viewport->makeTrgbColor(0x00, 0x00, 0x00, 180);
	viewDraw->setSymbology(dimLineColor, dimLineColor, 1, false);
	for (BPGraphics::EntryPtr& loadedEntry : *ptrGraphics)
	{
		switch (loadedEntry->getType())
		{
		case BPGraphics::Entry::Type::GeCurveBase:
		{
			IGeCurveBaseP pCurve = loadedEntry->getAsGeCurveBaseP();
			GeCurveArrayPtr ptrCv = GeCurveArray::create(pCurve, GeCurveArray::BOUNDARY_TYPE_Open);
			if (ptrCv.isValid())
			{
			}
		}
		break;
		case BPGraphics::Entry::Type::GeCurveArray:
		{
			GeCurveArrayP pCurveVector = loadedEntry->getAsGeCurveArrayP();
			if (nullptr != pCurveVector)
			{
			}
		}
		break;
		case BPGraphics::Entry::Type::Text:
		{
			BPTextPtr ptrText = loadedEntry->getAsTextEntP();
			if (ptrText.isValid())
			{
			}
		}
		default:
			break;
		}//end of switch
	}//end of for
}


MyCubeDragManipulatorDemo* MyCubeDragManipulatorDemo::Create()
{
	return new MyCubeDragManipulatorDemo();
}

BPIDragManipulatorP MyCubeDragManipulatorDemoExtension::_getIDragManipulator(::BIMBase::Core::BPEntityCR elHandle, ::BIMBase::Core::BPPickDataCP path)
{
	return MyCubeDragManipulatorDemo::Create();
}
