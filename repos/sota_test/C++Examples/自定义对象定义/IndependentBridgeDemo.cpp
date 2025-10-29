#include "stdafx.h"


using namespace DemoObject;
using namespace p3d;

#define PROPERTY_NAME              "name"
#define PROPERTY_PATTERN           "pattern"
#define PROPERTY_COLUMNDIAMETER    "columnDiameter"
#define PROPERTY_COLUMNHIGHT       "columnHight"
#define PROPERTY_BRIDGEARCHHIGHT   "bridgeArchHight"
#define PROPERTY_NUMROWS           "numRows"
#define PROPERTY_NUMCOLUMNS        "numColumns"
#define PROPERTY_CSSLONG           "CSSLong"
#define PROPERTY_CSSWIDTH          "CSSWidth"
#define PROPERTY_CSSHIGHT          "CSSHight"
#define PROPERTY_TOPSLABTHICKNESS  "topSlabThickness"
#define PROPERTY_SIDESLABTHICKNESS "sideSlabThickness"
#define PROPERTY_TUBES             "tubes"


IndependentBridge::IndependentBridge()
{
	m_wsName = L"QJ-独立桥架";
	m_enPattern = IBPattern::enArchBridge;

	TubePtr tube1 = new Tube(GePoint3d::create(0, 0, 150), 280, 10);
	TubePtr tube2 = new Tube(GePoint3d::create(-300, 0, 150), 280, 10);
	TubePtr tube3 = new Tube(GePoint3d::create(300, 0, 150), 280, 10);
	TubePtr tube4 = new Tube(GePoint3d::create(0, 0, 450), 280, 10);
	TubePtr tube5 = new Tube(GePoint3d::create(-300, 0, 450), 280, 10);
	TubePtr tube6 = new Tube(GePoint3d::create(300, 0, 450), 280, 10);


	m_gTubes.clear();
	m_gTubes.push_back(tube1);
	m_gTubes.push_back(tube2);
	m_gTubes.push_back(tube3);
	m_gTubes.push_back(tube4);
	m_gTubes.push_back(tube5);
	m_gTubes.push_back(tube6);

	m_dColumnDiameter = 1000;
	m_dColumnHight = 3000;
	m_dBridgeArchHight = 2500;
	m_nNumRows = 2;
	m_nNumColumns = 3;
	m_dCSSLong = 20000;
	m_dCSSWidth = 1000;
	m_dCSSHight = 1000;
	m_dTopSlabThickness = 50;
	m_dSideSlabThickness = 50;
}

IndependentBridge::~IndependentBridge()
{
	m_gTubes.clear();
}

void IndependentBridge::setName(CString wsName)
{
	m_wsName = wsName;
}

CString IndependentBridge::getName() const
{
	return m_wsName;
}

void IndependentBridge::setIBPattern(IBPattern enPattern)
{
	m_enPattern = enPattern;
}

IBPattern IndependentBridge::getIBPattern() const
{
	return m_enPattern;
}

void IndependentBridge::setTubes(std::vector<TubePtr> gTubes)
{
	m_gTubes = gTubes;
}

std::vector<TubePtr> IndependentBridge::getTubes() const
{
	return m_gTubes;
}

void IndependentBridge::setColumnDiameter(double dColumnDiameter)
{
	m_dColumnDiameter = dColumnDiameter;
}

double IndependentBridge::getColumnDiameter() const
{
	return m_dColumnDiameter;
}

void IndependentBridge::setColumnHight(double dColumnHight)
{
	m_dColumnHight = dColumnHight;
}

double IndependentBridge::getColumnHight() const
{
	return m_dColumnHight;
}

void IndependentBridge::setBridgeArchHight(double dBridgeArchHight)
{
	m_dBridgeArchHight = dBridgeArchHight;
}

double IndependentBridge::getBridgeArchHight() const
{
	return m_dBridgeArchHight;
}

void IndependentBridge::setNumRows(int nNumRows)
{
	m_nNumRows = nNumRows;
	__updateTubes();
}

void IndependentBridge::setNumColumns(int nNumColumns)
{
	m_nNumColumns = nNumColumns;
	__updateTubes();
}

int IndependentBridge::getNumColumns() const
{
	return m_nNumColumns;
}

int IndependentBridge::getNumRows() const
{
	return m_nNumRows;
}

void IndependentBridge::setCSSLong(double dCSSLong)
{
	m_dCSSLong = dCSSLong;
}

double IndependentBridge::getCSSLong() const
{
	return m_dCSSLong;
}

void IndependentBridge::setCSSWidth(double dCSSWidth)
{
	m_dCSSWidth = dCSSWidth;
}

double IndependentBridge::getCSSWidth() const
{
	return m_dCSSWidth;
}

void IndependentBridge::setCSSHight(double dCSSHight)
{
	m_dCSSHight = dCSSHight;
}

double IndependentBridge::getCSSHight() const
{
	return m_dCSSHight;
}

void IndependentBridge::setTopSlabThickness(double dTopSlabThickness)
{
	m_dTopSlabThickness = dTopSlabThickness;
}

double IndependentBridge::getTopSlabThickness() const
{
	return m_dTopSlabThickness;
}

void IndependentBridge::setSideSlabThickness(double dSideSlabThickness)
{
	m_dSideSlabThickness = dSideSlabThickness;
}

double IndependentBridge::getSideSlabThickness() const
{
	return m_dSideSlabThickness;
}

void IndependentBridge::setTubeDiameter(vector<double> dTubeDiameters)
{
	if (dTubeDiameters.size() != m_gTubes.size())
		return;

	for (int i = 0; i < dTubeDiameters.size(); i++)
	{
		m_gTubes.at(i)->setDiameter(dTubeDiameters.at(i));
	}
}

vector<double> IndependentBridge::getTubeDiameter() const
{
	vector<double> dias;
	for (auto tube : m_gTubes)
	{
		dias.push_back(tube->getDiameter());
	}
	return dias;
}

void IndependentBridge::setTubeThickness(vector<double> dTubeThickness)
{
	if (dTubeThickness.size() != m_gTubes.size())
		return;

	for (int i = 0; i < dTubeThickness.size(); i++)
	{
		m_gTubes.at(i)->setThickness(dTubeThickness.at(i));
	}
}

vector<double> IndependentBridge::getTubeThickness() const
{
	vector<double> thics;
	for (auto tube : m_gTubes)
	{
		thics.push_back(tube->getThickness());
	}
	return thics;
}

void DemoObject::IndependentBridge::setTubeCenters(pvector<GePoint3d> pts)
{
	int nPtsSize = pts.size();
	if (nPtsSize != m_gTubes.size())
		return;
	for (int i = 0; i < nPtsSize; i++)
	{
		m_gTubes.at(i)->setCenter(pts.at(i));
	}
}

pvector<GePoint3d> DemoObject::IndependentBridge::getTubeCenters()
{
	pvector<GePoint3d> pts;
	for (auto tube : m_gTubes)
	{
		pts.push_back(tube->getCenter());
	}
	return pts;
}

::p3d::P3DStatus DemoObject::IndependentBridge::_initFromData(BIMBase::Core::BPDataCR	instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, PROPERTY_NAME);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setName(value.getWString().c_str());


	status = instance.getValue(value, PROPERTY_PATTERN);
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	size_t nSize;
	const byte* constBytes = value.getBinary(nSize);
	byte* bytes = const_cast<byte*>(constBytes);
	int nTem;
	P3DStatus coverStatus = BPValue::getIntFromBytes(nTem, bytes, nSize);
	if (0 != coverStatus)
		return ERROR;
	setIBPattern((IBPattern)nTem);

	status = instance.getValue(value, PROPERTY_COLUMNDIAMETER);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setColumnDiameter(value.getDouble());

	status = instance.getValue(value, PROPERTY_COLUMNHIGHT);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setColumnHight(value.getDouble());

	status = instance.getValue(value, PROPERTY_BRIDGEARCHHIGHT);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setBridgeArchHight(value.getDouble());

	status = instance.getValue(value, PROPERTY_NUMROWS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setNumRows(value.getInteger());

	status = instance.getValue(value, PROPERTY_NUMCOLUMNS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setNumColumns(value.getInteger());

	status = instance.getValue(value, PROPERTY_CSSLONG);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setCSSLong(value.getDouble());

	status = instance.getValue(value, PROPERTY_CSSWIDTH);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setCSSWidth(value.getDouble());

	status = instance.getValue(value, PROPERTY_CSSHIGHT);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setCSSHight(value.getDouble());

	status = instance.getValue(value, PROPERTY_TOPSLABTHICKNESS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setTopSlabThickness(value.getDouble());

	status = instance.getValue(value, PROPERTY_SIDESLABTHICKNESS);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setSideSlabThickness(value.getDouble());

	status = instance.getValue(value, PROPERTY_TUBES);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	m_gTubes.clear();
	int arrCount = value.getArrayCount();
	for (int i = 0; i < arrCount; i++)
	{
		BPValue _value;

		status = instance.getValue(_value, PROPERTY_TUBES, i);
		if (P3DStatus::SUCCESS != status)
			return ERROR;
		BPDataPtr insPtr = _value.getStruct();

		if (insPtr == nullptr)
			return ERROR;
		TubePtr tubePtr = new Tube();
		P3DStatus _status = tubePtr->initFromData(*insPtr);
		if (0 != _status)
			return ERROR;
		m_gTubes.push_back(tubePtr);
	}

	return SUCCESS;
}

::p3d::P3DStatus DemoObject::IndependentBridge::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(PROPERTY_NAME, BPValue(getName()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	size_t stSize = sizeof((int)getIBPattern());
	byte* bytes = new byte[stSize];
	P3DStatus pstatus = BPDataUtil::setIntToBytes((int)getIBPattern(), bytes, stSize);
	if (P3DStatus::SUCCESS != pstatus)
		return ERROR;
	BPValue binaryValue;
	binaryValue.setBinary(bytes, stSize);
	status = instance.setValue(PROPERTY_PATTERN, binaryValue);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	if (bytes)
	{
		delete[] bytes;
		bytes = NULL;
	}

	status = instance.setValue(PROPERTY_COLUMNDIAMETER, BPValue(getColumnDiameter()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_COLUMNHIGHT, BPValue(getColumnHight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_BRIDGEARCHHIGHT, BPValue(getBridgeArchHight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_NUMROWS, BPValue(getNumRows()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_NUMCOLUMNS, BPValue(getNumColumns()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_CSSLONG, BPValue(getCSSLong()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_CSSWIDTH, BPValue(getCSSWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_CSSHIGHT, BPValue(getCSSHight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_TOPSLABTHICKNESS, BPValue(getTopSlabThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_SIDESLABTHICKNESS, BPValue(getSideSlabThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.clearArray(PROPERTY_TUBES);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	status = instance.addArrayElements(PROPERTY_TUBES, m_gTubes.size());
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	for (int i = 0; i < m_gTubes.size(); i++)
	{
		BPValue _value;
		BPDataPtr _ptrInstance = m_gTubes.at(i)->getData(project);
		if (_ptrInstance.isNull())
		{
			_ptrInstance = BPDataUtil::createDataByName(project, PBM_SCHEMA_Demo, "Tube", BPDataFlag::enNormal);
			if (_ptrInstance.isNull())
				continue;
			m_gTubes.at(i)->copyToData(*_ptrInstance, project);
		}

		if (_ptrInstance == nullptr)
			return ERROR;
		_value.setStruct(_ptrInstance.get());
		status = instance.setValue(PROPERTY_TUBES, _value, i);
		if (P3DStatus::SUCCESS != status)
			return ERROR;
	}
	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr IndependentBridge::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics)
{

#if 1
	BPModelPtr modelPtr = project.loadModelById(modelId);
	if (modelPtr == nullptr)
		return nullptr;

	//桥架，六面体,布尔运算
	GeBoxInfo boxOutterInfo(
		GePoint3d::createByZero(),
		GePoint3d::create(0, getCSSLong(), 0),
		GeVec3d::create(1, 0, 0),
		GeVec3d::create(0, 0, 1),
		getCSSWidth(),
		getCSSHight(),
		getCSSWidth(),
		getCSSHight(),
		true
	);
	IGeSolidBasePtr boxOutterPtr = IGeSolidBase::createGeBox(boxOutterInfo);
	if (boxOutterPtr == nullptr)
		return nullptr;
	if (!boxOutterPtr->transform(GeTransform::create(GePoint3d::create(-m_dCSSWidth / 2, 0, m_dColumnHight))))
		return nullptr;
	BPGraphicsPtr graphicOutterPtr = new BPGraphics(modelPtr);
	if (P3DStatus::SUCCESS != graphicOutterPtr->addGeSolidBase(*boxOutterPtr))
		return nullptr;

	GeBoxInfo boxInnerInfo(
		GePoint3d::create(getSideSlabThickness(), 0, 0),
		GePoint3d::create(getSideSlabThickness(), getCSSLong(), 0),
		GeVec3d::create(1, 0, 0),
		GeVec3d::create(0, 0, 1),
		getCSSWidth() - 2 * getSideSlabThickness(),
		getCSSHight() - getTopSlabThickness(),
		getCSSWidth() - 2 * getSideSlabThickness(),
		getCSSHight() - getTopSlabThickness(),
		true
	);
	IGeSolidBasePtr boxInnerPtr = IGeSolidBase::createGeBox(boxInnerInfo);
	if (boxInnerPtr == nullptr)
		return nullptr;
	if (!boxInnerPtr->transform(GeTransform::create(GePoint3d::create(-m_dCSSWidth / 2, 0, m_dColumnHight))))
		return nullptr;
	BPGraphicsPtr graphicInnerPtr = new BPGraphics(modelPtr);
	if (P3DStatus::SUCCESS != graphicInnerPtr->addGeSolidBase(*boxInnerPtr))
		return nullptr;

	BPGraphicsPtr graphicPtr = new BPGraphics(modelPtr);
	if (0 != BPSolidBooleanUtil::doBoolean(graphicPtr, graphicOutterPtr, graphicInnerPtr, BPBooleanOp::Substract))
		return nullptr;
	if (graphicPtr == nullptr)
		return nullptr;


	//桥墩,圆台体
	GeConeInfo coneInfo(GePoint3d::createByZero(),
		GePoint3d::create(0, 0, getColumnHight()),
		GeVec3d::create(1, 0, 0),
		GeVec3d::create(0, 1, 0),
		getColumnDiameter() / 2,
		getColumnDiameter() / 2,
		true
	);
	IGeSolidBasePtr columnStartPtr = IGeSolidBase::createGeCone(coneInfo);
	if (columnStartPtr == nullptr)
		return nullptr;
	if (P3DStatus::SUCCESS != graphicPtr->addGeSolidBase(*columnStartPtr))
		return nullptr;

	IGeSolidBasePtr columnEndPtr = columnStartPtr->deepClone();
	if (columnEndPtr == nullptr)
		return nullptr;

	if (!columnEndPtr->transform(GeTransform::create(GePoint3d::create(0, getCSSLong(), 0))))
		return nullptr;
	if (P3DStatus::SUCCESS != graphicPtr->addGeSolidBase(*columnEndPtr))
		return nullptr;


	//排管，拉身体
	GeCurveArrayPtr baseCurvePtr = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_ParityRegion);
	for (TubePtr tubePtr : m_gTubes)
	{
		GeEllipse3d ellipseOuter = GeEllipse3d::createByPoints(
			tubePtr->getCenter(),
			tubePtr->getCenter() + GePoint3d::create(tubePtr->getDiameter() / 2 + tubePtr->getThickness(), 0, 0),
			tubePtr->getCenter() + GePoint3d::create(0, 0, tubePtr->getDiameter() / 2 + tubePtr->getThickness()),
			0,
			360);
		IGeCurveBasePtr outterCuvePtr = IGeCurveBase::createEllipse(ellipseOuter);
		if (outterCuvePtr == nullptr)
			continue;
		GeCurveArrayPtr outterCurveArrayPtr = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Outer);
		outterCurveArrayPtr->add(outterCuvePtr);

		GeEllipse3d ellipseInner = GeEllipse3d::createByPoints(
			tubePtr->getCenter(),
			tubePtr->getCenter() + GePoint3d::create(tubePtr->getDiameter() / 2, 0, 0),
			tubePtr->getCenter() + GePoint3d::create(0, 0, tubePtr->getDiameter() / 2),
			0,
			360);
		IGeCurveBasePtr innerCuvePtr = IGeCurveBase::createEllipse(ellipseInner);
		if (innerCuvePtr == nullptr)
			continue;
		GeCurveArrayPtr innerCurveArrayPtr = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Inner);
		innerCurveArrayPtr->add(innerCuvePtr);

		baseCurvePtr->add(outterCurveArrayPtr);
		baseCurvePtr->add(innerCurveArrayPtr);
	}

	GeExtrusionInfo tubeInfo(
		baseCurvePtr,
		GeVec3d::create(GePoint3d::create(0, m_dCSSLong, 0)),
		true
	);
	IGeSolidBasePtr tubeSolidPtr = IGeSolidBase::createGeExtrusion(tubeInfo);
	if (tubeSolidPtr == nullptr)
		return nullptr;
	if (!tubeSolidPtr->transform(GeTransform::create(GePoint3d::create(0, 0, m_dColumnHight))))
		return nullptr;
	if (P3DStatus::SUCCESS != graphicPtr->addGeSolidBase(*tubeSolidPtr))
		return nullptr;


	//桁架，拉伸体
	double dGabarit = m_dCSSLong - m_dColumnDiameter;
	//净跨榀数，取偶数
	if (0 == m_dCSSHight)
		return nullptr;
	int nNum = 2 * std::max((int)(dGabarit / (4 * m_dCSSHight)), 1);
	double dSingleSpan = dGabarit / nNum;
	int nTrussWidhtModelNum = std::max(dSingleSpan / 100, 1.0);
	int nTrussHight = 5 * nTrussWidhtModelNum;
	int nTrussWidth = nTrussHight / 4;

	//U型杆，拉伸体
	GeCurveArrayPtr caUPtr = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> pts = {
		GePoint3d::create(-m_dCSSWidth / 2 - nTrussWidth, 0, m_dCSSHight),
		GePoint3d::create(-m_dCSSWidth / 2 - nTrussWidth, 0, -nTrussWidth),
		GePoint3d::create(m_dCSSWidth / 2 + nTrussWidth, 0, -nTrussWidth),
		GePoint3d::create(m_dCSSWidth / 2 + nTrussWidth, 0, m_dCSSHight),
		GePoint3d::create(m_dCSSWidth / 2, 0, m_dCSSHight),
		GePoint3d::create(m_dCSSWidth / 2, 0, 0),
		GePoint3d::create(-m_dCSSWidth / 2, 0, 0),
		GePoint3d::create(-m_dCSSWidth / 2, 0, m_dCSSHight)
	};
	IGeCurveBasePtr cbUPtr = IGeCurveBase::createLineString(pts);
	if (cbUPtr.isNull())
		return nullptr;
	caUPtr->add(cbUPtr);

	GeExtrusionInfo uExtrusionInfo(
		caUPtr,
		GeVec3d::create(GePoint3d::create(0, nTrussHight, 0)),
		true
	);
	IGeSolidBasePtr uTrussPtr = IGeSolidBase::createGeExtrusion(uExtrusionInfo);
	if (uTrussPtr.isNull())
		return nullptr;

	double dSingleSpanForUTruss = (dGabarit - nTrussHight) / nNum;
	for (int i = 0; i < nNum + 1; i++)
	{
		IGeSolidBasePtr uTrussTemPtr = uTrussPtr->deepClone();
		if (!uTrussTemPtr->transform(GeTransform::create(GeVec3d::create(GePoint3d::create(0, i * dSingleSpanForUTruss + m_dColumnDiameter / 2, m_dColumnHight)))))
			continue;;
		graphicPtr->addGeSolidBase(*uTrussTemPtr);
	}


	if (IBPattern::enParallelBridge == m_enPattern)
	{
		//端部斜杆
		GePoint3d ptStart = GePoint3d::create(m_dCSSWidth / 2, 0, m_dColumnHight);
		GePoint3d ptEnd = GePoint3d::create(m_dCSSWidth / 2, m_dColumnDiameter / 2, m_dColumnHight + m_dCSSHight);

		GeVec3d vec = GeVec3d::createByRotate90Around(
			GeVec3d::createByStartEnd(ptEnd, ptStart),
			GeVec3d::createByStartEnd(ptEnd, ptEnd + GePoint3d::create(1, 0, 0)));
		vec.normalize();
		vec *= nTrussHight;

		GePoint3d ptStartOffset = GePoint3d::createByTransform(GeTransform::create(vec), ptStart);
		GePoint3d ptEndOffset = GePoint3d::createByTransform(GeTransform::create(vec), ptEnd);

		pvector<GePoint3d> pts = { ptStart, ptStartOffset, ptEndOffset, ptEnd };
		GeCurveArrayPtr _caPtr = GeCurveArray::createLinestringArray(pts, GeCurveArray::BOUNDARY_TYPE_Outer);
		if (_caPtr.isNull())
			return nullptr;

		GeExtrusionInfo headExtrusionInfo(
			_caPtr,
			GeVec3d::create(GePoint3d::create(nTrussWidth, 0, 0)),
			true
		);
		IGeSolidBasePtr headTrussPtr = IGeSolidBase::createGeExtrusion(headExtrusionInfo);
		if (headTrussPtr.isNull())
			return nullptr;

		graphicPtr->addGeSolidBase(*headTrussPtr);


		GeTransform minrrorTrans;
		if (!minrrorTrans.setByMirrorPlane(GePoint3d::create(0, m_dCSSLong / 2, 0), GeVec3d::create(0, 1, 0)))
			return nullptr;

		IGeSolidBasePtr endTrussPtr = headTrussPtr->deepClone();
		if (endTrussPtr.isNull())
			return nullptr;

		if (!endTrussPtr->transform(minrrorTrans))
			return nullptr;

		graphicPtr->addGeSolidBase(*endTrussPtr);


		IGeSolidBasePtr headTrussOSPtr = headTrussPtr->deepClone();
		if (headTrussOSPtr.isNull())
			return nullptr;

		if (!headTrussOSPtr->transform(GeTransform::create(GePoint3d::create(-m_dCSSWidth - nTrussWidth, 0, 0))))
			return nullptr;
		graphicPtr->addGeSolidBase(*headTrussOSPtr);

		IGeSolidBasePtr endTrussOSPtr = endTrussPtr->deepClone();
		if (endTrussOSPtr.isNull())
			return nullptr;

		if (!endTrussOSPtr->transform(GeTransform::create(GePoint3d::create(-m_dCSSWidth - nTrussWidth, 0, 0))))
			return nullptr;
		graphicPtr->addGeSolidBase(*endTrussOSPtr);

		//桁架斜杆
		GePoint3d ptTR = GePoint3d::create(m_dCSSWidth / 2, m_dColumnDiameter / 2 + nTrussHight / 2, m_dColumnHight + m_dCSSHight);
		GePoint3d ptBR = GePoint3d::create(m_dCSSWidth / 2, m_dColumnDiameter / 2 + nTrussHight / 2 + dSingleSpanForUTruss, m_dColumnHight);

		GeVec3d vecSolpeTruss = GeVec3d::createByRotate90Around(
			GeVec3d::createByStartEnd(ptBR, ptTR),
			GeVec3d::createByStartEnd(ptBR, ptTR + GePoint3d::create(1, 0, 0)));
		vecSolpeTruss.normalize();
		vecSolpeTruss *= nTrussHight;

		GePoint3d ptTL = GePoint3d::createByTransform(GeTransform::create(vecSolpeTruss), ptTR);
		GePoint3d ptBL = GePoint3d::createByTransform(GeTransform::create(vecSolpeTruss), ptBR);

		pvector<GePoint3d> slopePts = { ptTR, ptTL, ptBL, ptBR };
		GeCurveArrayPtr _slopeCaPtr = GeCurveArray::createLinestringArray(slopePts, GeCurveArray::BOUNDARY_TYPE_Outer);
		if (_slopeCaPtr.isNull())
			return nullptr;
		GeExtrusionInfo slopeTrussPtotoTypeInfo(
			_slopeCaPtr,
			GeVec3d::create(GePoint3d::create(nTrussWidth, 0, 0)),
			true
		);
		IGeSolidBasePtr slopeTrussPtotoTypePtr = IGeSolidBase::createGeExtrusion(slopeTrussPtotoTypeInfo);
		if (slopeTrussPtotoTypePtr.isNull())
			return nullptr;

		vector<IGeSolidBasePtr> trussPair = { slopeTrussPtotoTypePtr };
		IGeSolidBasePtr slopeTrussPtotoTypePaterPtr = slopeTrussPtotoTypePtr->deepClone();
		if (slopeTrussPtotoTypePaterPtr.isNull())
			return nullptr;
		GeTransform slopePTMirror;
		if (!slopePTMirror.setByMirrorPlane(GePoint3d::create(0, m_dColumnDiameter / 2 + nTrussHight / 2 + dSingleSpanForUTruss, 0), GeVec3d::create(0, 1, 0)))
			return nullptr;
		if (!slopeTrussPtotoTypePaterPtr->transform(slopePTMirror))
			return nullptr;
		trussPair.push_back(slopeTrussPtotoTypePaterPtr);

		for (int i = 0; i < nNum / 2; i++)
		{
			for (int j = 0; j < trussPair.size(); j++)
			{
				IGeSolidBasePtr _truss = trussPair.at(j)->deepClone();
				if (_truss.isNull())
					continue;
				if (!_truss->transform(GeTransform::create(GePoint3d::create(0, i * dSingleSpanForUTruss * 2, 0))))
					continue;
				graphicPtr->addGeSolidBase(*_truss);

				IGeSolidBasePtr _trussOS = _truss->deepClone();
				if (_trussOS.isNull())
					continue;
				if (!_trussOS->transform(GeTransform::create(GePoint3d::create(-m_dCSSWidth - nTrussWidth, 0, 0))))
					continue;
				graphicPtr->addGeSolidBase(*_trussOS);
			}
		}

	}
	else
	{
		//弧拱
		if (!__createBridteArch(graphicPtr, nNum, dSingleSpanForUTruss, nTrussHight))
			return nullptr;
	}
	
	return graphicPtr;
#endif

}

void IndependentBridge::__mirror(BPGraphicsPtr& graphicsPtr, IGeSolidBasePtr originSolidPtr, GeTransform mirrorYZ, GeTransform mirrorXZ)
{
	if (graphicsPtr.isNull())
		return;
	if (originSolidPtr.isNull())
		return;

	//横向镜像
	IGeSolidBasePtr wMirrorPtr = originSolidPtr->deepClone();
	if (wMirrorPtr.isNull())
		return;
	if (!wMirrorPtr->transform(mirrorYZ))
		return;
	graphicsPtr->addGeSolidBase(*wMirrorPtr);

	//跨向镜像
	IGeSolidBasePtr lMirrorPtr = originSolidPtr->deepClone();
	if (lMirrorPtr.isNull())
		return;
	if (!lMirrorPtr->transform(mirrorXZ))
		return;
	graphicsPtr->addGeSolidBase(*lMirrorPtr);

	//横向镜像 再跨向镜像
	IGeSolidBasePtr wlMirrorPtr = wMirrorPtr->deepClone();
	if (wlMirrorPtr.isNull())
		return;
	if (!wlMirrorPtr->transform(mirrorXZ))
		return;
	graphicsPtr->addGeSolidBase(*wlMirrorPtr);

	return;
}

void IndependentBridge::__updateTubes()
{
	vector<TubePtr> tubeRows;
	GePoint3d tubeCenter = GePoint3d::createByZero();
	double tubeDiameter = 280, tubeThichness = 10;
	if (0 != m_gTubes.size())
	{
		tubeDiameter = m_gTubes.at(0)->getDiameter();
		tubeThichness = m_gTubes.at(0)->getThickness();
	}

	m_gTubes.clear();
	for (int i = 0; i < m_nNumColumns; i++)
	{
		TubePtr tubePtr = new Tube();
		tubePtr->setThickness(tubeThichness);
		tubePtr->setDiameter(tubeDiameter);
		tubeCenter = GePoint3d::create((tubeDiameter + 2 * tubeThichness) * (m_nNumColumns / 2.0 - i - 0.5), 0, (tubeDiameter + 2 * tubeThichness) / 2.0);
		tubePtr->setCenter(tubeCenter);
		tubeRows.push_back(tubePtr);
		m_gTubes.push_back(tubePtr);
	}

	if (m_nNumRows < 2)
		return;

	for (int i = 0; i < m_nNumRows - 1; i++)
	{
		for (auto tube : tubeRows)
		{
			TubePtr tubePtr = tube->deepClone();
			if (tubePtr.isNull())
				continue;

			tubePtr->setCenter(tube->getCenter() + GePoint3d::create(0, 0, (tubeDiameter + 2 * tubeThichness) * (i + 1)));
			m_gTubes.push_back(tubePtr);
		}
	}

	//调整桥架宽度和高度
	m_dCSSWidth = m_nNumColumns * (tubeDiameter + 2 * tubeThichness) + m_dSideSlabThickness * 2;
	if (m_dCSSHight < m_nNumRows * (tubeDiameter + 2 * tubeThichness) + m_dTopSlabThickness)
		m_dCSSHight = m_nNumRows * (tubeDiameter + 2 * tubeThichness) + m_dTopSlabThickness;
}


bool ::IndependentBridge::__createBridteArch(BPGraphicsPtr& graphicsPtr, int nNum, double dSingleSpanForUTruss, int nTrussHight)
{
	if (m_dBridgeArchHight <= 0)
		return false;
	double dR = (pow(m_dCSSLong / 2 - m_dColumnDiameter / 4, 2) + pow(m_dBridgeArchHight, 2)) / (2 * m_dBridgeArchHight);
	if (dR <= 0)
		return false;
	double dArchTubeDiameter = min((m_dCSSLong - m_dColumnDiameter / 2) / 40, m_dCSSWidth / 6);

	GeCurveArrayPtr outerPtr = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	GeVec3d vecR = GeVec3d::createByStartEnd(
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dColumnDiameter / 4, m_dColumnHight + m_dCSSHight),
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dCSSLong / 2, m_dColumnHight + m_dBridgeArchHight - dR + m_dCSSHight));
	GeVec3d vecTangent = GeVec3d::createByRotate90Around(vecR, GeVec3d::create(1, 0, 0));

	GeEllipse3d cirlceOuter = GeEllipse3d::createByCircle(
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dColumnDiameter / 4, m_dColumnHight + m_dCSSHight),
		vecTangent,
		dArchTubeDiameter / 2);
	IGeCurveBasePtr cbPtr = IGeCurveBase::createEllipse(cirlceOuter);
	outerPtr->add(cbPtr);

	//内角
	double dAngle = asin((m_dCSSLong / 2 - m_dColumnDiameter / 4) / dR);
	//拱
	GeRotationalSweepInfo info(
		outerPtr,
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dCSSLong / 2, m_dColumnHight + m_dCSSHight + m_dBridgeArchHight - dR),
		GeVec3d::create(1, 0, 0),
		-dAngle * 2,
		true
	);

	IGeSolidBasePtr bridgeArchPtr = IGeSolidBase::createGeRotationalSweep(info);
	if (bridgeArchPtr.isNull())
		return false;
	if (0 != graphicsPtr->addGeSolidBase(*bridgeArchPtr))
		return false;

	IGeSolidBasePtr bridgeArchOptr = bridgeArchPtr->deepClone();
	if (bridgeArchOptr.isNull())
		return false;

	if (!bridgeArchOptr->transform(GeTransform::create(GePoint3d::create(-m_dCSSWidth + dArchTubeDiameter, 0, 0))))
		return false;

	if (0 != graphicsPtr->addGeSolidBase(*bridgeArchOptr))
		return false;


	//桁架拱
	//拱基线
	GeEllipse3d archBaseline = GeEllipse3d::createByPointsOnEllipse(
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dCSSLong - m_dColumnDiameter / 4, m_dColumnHight + m_dCSSHight),
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dCSSLong / 2, m_dColumnHight + m_dCSSHight + m_dBridgeArchHight),
		GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, m_dColumnDiameter / 4, m_dColumnHight + m_dCSSHight)
	);
	IGeCurveBasePtr cbArchBaselinePtr = IGeCurveBase::createEllipse(archBaseline);

	vector<GePoint3d> ptsStart;
	vector<GePoint3d> ptsIntersect;
	GeTransform mirrorXZ, mirrorYZ;
	if (!mirrorXZ.setByMirrorPlane(GePoint3d::create(0, m_dCSSLong / 2, 0), GeVec3d::create(0, 1, 0)))
		return false;
	if (!mirrorYZ.setByMirrorPlane(GePoint3d::createByZero(), GeVec3d::create(1, 0, 0)))
		return false;

	//竖杆
	for (int i = 0; i < nNum / 2 + 1; i++)
	{
		GePoint3d ptStart = GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, i * dSingleSpanForUTruss + m_dColumnDiameter / 2 + nTrussHight / 2, m_dColumnHight + m_dCSSHight);

		GeSegment3d segment = GeSegment3d::create(
			GePoint3d::create(m_dCSSWidth / 2 - dArchTubeDiameter / 2, i * dSingleSpanForUTruss + m_dColumnDiameter / 2 + nTrussHight / 2, m_dColumnHight),
			ptStart
		);

		GePoint3d pts[2], ellipseParams[2];
		double pLineParams[2];
		int nPtNum = archBaseline.getIntersectPointsWithSegment(pts, ellipseParams, pLineParams, segment);

		GePoint3d ptEnd = GePoint3d::createByZero();
		for (GePoint3d mem : pts)
		{
			if (mem.z > m_dColumnHight + m_dCSSHight)
			{
				ptEnd = mem;
				ptsStart.push_back(ptStart);
				ptsIntersect.push_back(mem);
			}
		}

		//竖杆
		GeConeInfo coneInfo(
			ptStart,
			ptEnd,
			dArchTubeDiameter / 2 * 0.75,
			dArchTubeDiameter / 2 * 0.75,
			true
		);
		IGeSolidBasePtr montantPtr = IGeSolidBase::createGeCone(coneInfo);
		if (montantPtr.isNull())
			continue;

		if (0 != graphicsPtr->addGeSolidBase(*montantPtr))
			continue;

		//横向镜像
		IGeSolidBasePtr montantWMirrorPtr = montantPtr->deepClone();
		if (montantWMirrorPtr.isNull())
			continue;
		if (!montantWMirrorPtr->transform(mirrorYZ))
			continue;
		graphicsPtr->addGeSolidBase(*montantWMirrorPtr);

		//中间竖杆不做镜像
		if (i == nNum / 2)
			continue;

		//跨向镜像
		IGeSolidBasePtr montantLMirrorPtr = montantPtr->deepClone();
		if (montantLMirrorPtr.isNull())
			continue;
		if (!montantLMirrorPtr->transform(mirrorXZ))
			continue;
		graphicsPtr->addGeSolidBase(*montantLMirrorPtr);

		//横向镜像 再跨向镜像
		IGeSolidBasePtr montantWLMirrorPtr = montantWMirrorPtr->deepClone();
		if (montantWLMirrorPtr.isNull())
			continue;
		if (!montantWLMirrorPtr->transform(mirrorXZ))
			continue;
		graphicsPtr->addGeSolidBase(*montantWLMirrorPtr);
	}

	//水平支撑
	int nSpportSize = ptsStart.size();
	if (nSpportSize < 2)
		return true;
	for (int i = 0; i < nSpportSize - 1; i++)
	{
		GePoint3d ptS = ptsStart.at(i);
		GePoint3d ptE = ptsIntersect.at(i);
		GePoint3d ptSp = ptsStart.at(i + 1);
		GePoint3d ptEp = ptsIntersect.at(i + 1);
		GePoint3d ptEpMirror = GePoint3d::createByTransform(mirrorYZ, ptEp);

		//水平斜向支撑
		GeConeInfo sopportHSInfo(
			ptE,
			ptEpMirror,
			dArchTubeDiameter / 2 * 0.5,
			dArchTubeDiameter / 2 * 0.5,
			true
		);
		IGeSolidBasePtr spportHSPtr = IGeSolidBase::createGeCone(sopportHSInfo);
		if (spportHSPtr.isNull())
			continue;
		if (0 != graphicsPtr->addGeSolidBase(*spportHSPtr))
			continue;
		__mirror(graphicsPtr, spportHSPtr, mirrorYZ, mirrorXZ);

		//水平横向支撑
		if (i == nSpportSize - 2)
		{
			GeConeInfo sopportHInfo(
				ptEp,
				ptEpMirror,
				dArchTubeDiameter / 2,
				dArchTubeDiameter / 2,
				true
			);
			IGeSolidBasePtr spportHPtr = IGeSolidBase::createGeCone(sopportHInfo);
			if (spportHPtr.isNull())
				continue;
			if (0 != graphicsPtr->addGeSolidBase(*spportHPtr))
				continue;
		}

		//竖向支撑
		if (ptS.distance(ptE) < 500)
			continue;

		GeConeInfo info(
			ptE,
			ptSp,
			dArchTubeDiameter / 2 * 0.5,
			dArchTubeDiameter / 2 * 0.5,
			true
		);
		IGeSolidBasePtr spportVPtr = IGeSolidBase::createGeCone(info);
		if (spportVPtr.isNull())
			continue;
		if (0 != graphicsPtr->addGeSolidBase(*spportVPtr))
			continue;

		__mirror(graphicsPtr, spportVPtr, mirrorYZ, mirrorXZ);
	}
	return true;
}


AutoDoRegisterFunctionsBegin
CString name = L"IndependentBridge";
BPObjectExtensionManager::getInstance().registerBPObjectExtension(PBM_SCHEMA_Demo, static_cast<Utf8String>(name), new IndependentBridgeExtension());
AutoDoRegisterFunctionsEnd
