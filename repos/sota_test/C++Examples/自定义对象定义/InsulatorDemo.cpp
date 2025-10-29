#include "stdafx.h"
#include "InsulatorDemo.h"

using namespace DemoObject;

#include "stdafx.h"


InsulatorDemo::InsulatorDemo()
{
	m_nN = 2;
	/**单串绝缘子片数量*/
	m_nN1 = 15;
	/**绝缘子单片连接高度*/
	m_dH1 = 100;
	/**大伞裙半径*/
	m_dR1 = 50;
	/**小伞裙半径*/
	m_dR2 = 40;
	/**绝缘子串半径*/
	m_dR = 30;
	/**双串间距*/
	m_dD = 110;
	/**前端长度（构架端*/
	m_dFL = 100;
	/**后端长度（导线端）*/
	m_dAL = 50;
	
	m_nRed = 91;
	m_nGreen = 58;
	m_nBlue = 41;

	m_dAlpha = 0;
	m_vtAxisX = GeVec3d::create(1, 0, 0);
	m_vtAxisY = GeVec3d::create(0, 1, 0);
	m_vtAxisZ = GeVec3d::create(0, 0, 1);
}

InsulatorDemo::~InsulatorDemo()
{
}

int  InsulatorDemo::getN() const
{
	return m_nN;
}

void  InsulatorDemo::setN(const int N)
{
	m_nN = N;
}

double  InsulatorDemo::getD() const
{
	return m_dD;
}

void  InsulatorDemo::setD(const double D)
{
	m_dD = D;
}

int  InsulatorDemo::getN1() const
{
	return m_nN1;
}

void  InsulatorDemo::setN1(const int N1)
{
	m_nN1 = N1;
}

double  InsulatorDemo::getH1() const
{
	return m_dH1;
}

void  InsulatorDemo::setH1(const double H1)
{
	m_dH1 = H1;
}

double  InsulatorDemo::getR1() const
{
	return m_dR1;
}

void  InsulatorDemo::setR1(const double R1)
{
	m_dR1 = R1;
}

double  InsulatorDemo::getR2() const
{
	return m_dR2;
}

void  InsulatorDemo::setR2(const double R2)
{
	m_dR2 = R2;
}

double  InsulatorDemo::getR() const
{
	return m_dR;
}

void  InsulatorDemo::setR(const double R)
{
	m_dR = R;
}

double  InsulatorDemo::getFL() const
{
	return m_dFL;
}

void  InsulatorDemo::setFL(const double FL)
{
	m_dFL = FL;
}

double  InsulatorDemo::getAL() const
{
	return m_dAL;
}

void  InsulatorDemo::setAL(const double AL)
{
	m_dAL = AL;
}

void InsulatorDemo::setCenter(const GePoint3d ptCenter) {
	m_ptCenter = ptCenter;
}


::p3d::P3DStatus InsulatorDemo::_copyToData(::BIMBase::Core::BPDataR instance, ::BIMBase::Core::BPProjectR project) const
{
	if (__super::_copyToData(instance, project) != SUCCESS)
		return ERROR;
	if (SUCCESS != instance.setValue("ptCenter", BPValue(m_ptCenter)))
		return ERROR;
	if (SUCCESS != instance.setValue("N", BPValue(m_nN)))
		return ERROR;
	if (SUCCESS != instance.setValue("D", BPValue(m_dD)))
		return ERROR;
	if (SUCCESS != instance.setValue("N1", BPValue(m_nN1)))
		return ERROR;
	if (SUCCESS != instance.setValue("H1", BPValue(m_dH1)))
		return ERROR;
	if (SUCCESS != instance.setValue("R1", BPValue(m_dR1)))
		return ERROR;
	if (SUCCESS != instance.setValue("R2", BPValue(m_dR2)))
		return ERROR;
	if (SUCCESS != instance.setValue("R", BPValue(m_dR)))
		return ERROR;
	if (SUCCESS != instance.setValue("FL", BPValue(m_dFL)))
		return ERROR;
	if (SUCCESS != instance.setValue("AL", BPValue(m_dAL)))
		return ERROR;
	
	return SUCCESS;
}
::p3d::P3DStatus InsulatorDemo::_initFromData(::BIMBase::Core::BPDataCR instance)
{
	if (__super::_initFromData(instance) != SUCCESS)
		return ERROR;

	BPValue ecValue;

	if (SUCCESS != instance.getValue(ecValue, "ptCenter"))
		return ERROR;
	m_ptCenter = ecValue.getPoint3D();
	if (SUCCESS != instance.getValue(ecValue, "N"))
		return ERROR;
	m_nN = ecValue.getInteger();
	if (SUCCESS != instance.getValue(ecValue, "D"))
		return ERROR;
	m_dD = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "N1"))
		return ERROR;
	m_nN1 = ecValue.getInteger();
	if (SUCCESS != instance.getValue(ecValue, "H1"))
		return ERROR;
	m_dH1 = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "R1"))
		return ERROR;
	m_dR1 = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "R2"))
		return ERROR;
	m_dR2 = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "R"))
		return ERROR;
	m_dR = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "FL"))
		return ERROR;
	m_dFL = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "AL"))
		return ERROR;
	m_dAL = ecValue.getDouble();
	if (SUCCESS != instance.getValue(ecValue, "Temp"))
		return ERROR;
	m_temp = ecValue.getLong();

	return SUCCESS;
}

BPGraphicsPtr InsulatorDemo::_createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool isDynamics)
{
	BPModelP modelP = project.loadModelById(modelId);
	if (NULL == modelP)
		return nullptr;

	BPGraphicsPtr graphicsPtr = modelP->createPhysicalGraphics();
	if (!graphicsPtr.isValid())
		return nullptr;

	BIMBase::BPSymbology sym = BPGraphics::getDefaultSymbology();
	BPRgbaColor _p3dColor;
	_p3dColor.alpha = 0;
	_p3dColor.red = m_nRed;
	_p3dColor.blue = m_nBlue;
	_p3dColor.green = m_nGreen;
	sym.color = BPColorUtil::getEntityColor(_p3dColor, project, true);

	BIMBase::BPSymbology sym1 = BPGraphics::getDefaultSymbology();
	BPRgbaColor _p3dColor1;
	_p3dColor1.alpha = 0;
	_p3dColor1.red = 138;
	_p3dColor1.blue = 149;
	_p3dColor1.green = 151;
	sym1.color = BPColorUtil::getEntityColor(_p3dColor1, project, true);

	// 前端法兰
	GeCurveArrayPtr frontOuter = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	GePoint3d  frontStart = m_ptCenter + m_vtAxisX * m_dFL * 0.4 + m_vtAxisY * m_dR / 2 - m_vtAxisZ * m_dR * 0.1;//前端法兰下底面右角点
	frontOuter->add(IGeCurveBase::createSegment(
		GeSegment3d::create(
			frontStart,
			frontStart - m_vtAxisX * m_dFL * 0.4
		)));//上边线
	frontOuter->add(IGeCurveBase::createEllipse(
		GeEllipse3d::createByPointsOnEllipse(
			m_ptCenter + m_vtAxisY * m_dR / 2 - m_vtAxisZ * m_dR * 0.1, //起点
			m_ptCenter - m_vtAxisX * m_dR / 2 - m_vtAxisZ * m_dR * 0.1,//中点
			m_ptCenter - m_vtAxisY * m_dR / 2 - m_vtAxisZ * m_dR * 0.1//终点
		)));//圆弧
	frontOuter->add(IGeCurveBase::createSegment(
		GeSegment3d::create(
			m_ptCenter - m_vtAxisY * m_dR / 2 - m_vtAxisZ * m_dR * 0.1,
			frontStart - m_vtAxisY * m_dR)
	));//下边线
	frontOuter->add(IGeCurveBase::createSegment(
		GeSegment3d::create(
			frontStart - m_vtAxisY * m_dR,
			frontStart)
	));//右边线

	GeCurveArrayPtr frontCurve = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_ParityRegion);
	frontCurve->add(frontOuter);

	GeCurveArrayPtr frontInner = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Inner);
	frontInner->add(IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(m_ptCenter - m_vtAxisZ * m_dR * 0.1, -m_vtAxisZ, m_dR / 4)));
	frontCurve->add(frontInner);

	GeExtrusionInfo frontHeadInfo(frontCurve, m_vtAxisZ * m_dR * 0.2, true);
	IGeSolidBasePtr frontHead = IGeSolidBase::createGeExtrusion(frontHeadInfo);
	graphicsPtr->addGeSolidBase(*frontHead, sym1, m_dAlpha);

	double dYStart = (m_nN - 1) * m_dD / 2;
	double h = m_dH1 / 4.0;//每一小段的高度
	GePoint3d barStart = m_ptCenter + m_vtAxisX * 0.6 * m_dFL;//前端棒棒起点
	GePoint3d insulatorStart = m_ptCenter + m_vtAxisX * m_dFL;//串起点
	GePoint3d backStart = m_ptCenter + m_vtAxisX * (m_nN1 * m_dH1 + m_dFL) - m_vtAxisZ * m_dR * 0.1;//后端法兰起点

	//瓷套+前端棒棒+后端法兰
	for (int n = 0; n < m_nN; n++)//遍历每一联
	{
		//前端过渡段
		GeBoxInfo BoxInfo(
			m_ptCenter + m_vtAxisX * m_dFL * 0.4 - m_vtAxisY * (m_dD * (m_nN - 1) / 2.0 + m_dR * 0.1) - m_vtAxisZ * m_dR * 0.1,
			m_ptCenter + m_vtAxisX * m_dFL * 0.4 - m_vtAxisY * (m_dD * (m_nN - 1) / 2.0 + m_dR * 0.1) + m_vtAxisZ * m_dR * 0.1,
			m_vtAxisX,
			m_vtAxisY,
			m_dFL * 0.2, (m_nN - 1) * m_dD + m_dR * 0.2,
			m_dFL * 0.2, (m_nN - 1) * m_dD + m_dR * 0.2,
			true);
		IGeSolidBasePtr BoxPtr = IGeSolidBase::createGeBox(BoxInfo);
		graphicsPtr->addGeSolidBase(*BoxPtr, sym1, m_dAlpha);

		GeTransform trans = GeTransform::create(0, dYStart - n * m_dD, 0);//每串平移矩阵

		//前端棒棒
		IGeSolidBasePtr LinkBarPtr;
		GeBoxInfo LinkBarInfo(
			barStart - m_vtAxisY * m_dR * 0.1 - m_vtAxisZ * m_dR * 0.1,
			barStart - m_vtAxisY * m_dR * 0.1 + m_vtAxisZ * m_dR * 0.1,
			m_vtAxisX,
			m_vtAxisY,
			m_dFL * 0.4, m_dR * 0.2,
			m_dFL * 0.4, m_dR * 0.2,
			true);
		LinkBarPtr = IGeSolidBase::createGeBox(LinkBarInfo);
		LinkBarPtr->transform(trans);
		graphicsPtr->addGeSolidBase(*LinkBarPtr, sym1, m_dAlpha);

		//瓷套
		for (int i = 0; i < m_nN1; i++)
		{
			GeConeInfo ConeInfo1(
				insulatorStart + m_vtAxisX * h * (4 * i),
				insulatorStart + m_vtAxisX * h * (4 * i + 1),
				m_dR, m_dR, true);//第1段
			IGeSolidBasePtr ConePtr1 = IGeSolidBase::createGeCone(ConeInfo1);
			ConePtr1->transform(trans);
			graphicsPtr->addGeSolidBase(*ConePtr1, sym, m_dAlpha);

			GeConeInfo ConeInfo2(
				insulatorStart + m_vtAxisX * h * (4 * i + 1),
				insulatorStart + m_vtAxisX * h * (4 * i + 2),
				m_dR, m_dR1, true);//大伞群
			IGeSolidBasePtr ConePtr2 = IGeSolidBase::createGeCone(ConeInfo2);
			ConePtr2->transform(trans);
			graphicsPtr->addGeSolidBase(*ConePtr2, sym, m_dAlpha);

			GeConeInfo ConeInfo3(
				insulatorStart + m_vtAxisX * h * (4 * i + 2),
				insulatorStart + m_vtAxisX * h * (4 * i + 3),
				m_dR, m_dR, true);//第3段
			IGeSolidBasePtr ConePtr3 = IGeSolidBase::createGeCone(ConeInfo3);
			ConePtr3->transform(trans);
			graphicsPtr->addGeSolidBase(*ConePtr3, sym, m_dAlpha);

			GeConeInfo ConeInfo4(
				insulatorStart + m_vtAxisX * h * (4 * i + 3),
				insulatorStart + m_vtAxisX * h * (4 * i + 4),
				m_dR, m_dR2, true);//小伞群
			IGeSolidBasePtr ConePtr4 = IGeSolidBase::createGeCone(ConeInfo4);
			ConePtr4->transform(trans);
			graphicsPtr->addGeSolidBase(*ConePtr4, sym, m_dAlpha);
		}

		// 后端法兰面-准备
		GeCurveArrayPtr backCurve = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_ParityRegion);
		GeCurveArrayPtr backOuter = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
		backOuter->add(IGeCurveBase::createSegment(
			GeSegment3d::create(
				backStart - m_vtAxisY * m_dR / 2,
				backStart - m_vtAxisY * m_dR / 2 + m_vtAxisX * m_dAL
			)));
		backOuter->add(IGeCurveBase::createEllipse(
			GeEllipse3d::createByPointsOnEllipse(
				backStart - m_vtAxisY * m_dR / 2 + m_vtAxisX * m_dAL,
				backStart + m_vtAxisX * (m_dAL + m_dR / 2),
				backStart + m_vtAxisY * m_dR / 2 + m_vtAxisX * m_dAL
			)));
		backOuter->add(IGeCurveBase::createSegment(
			GeSegment3d::create(
				backStart + m_vtAxisY * m_dR / 2 + m_vtAxisX * m_dAL,
				backStart + m_vtAxisY * m_dR / 2)
		));
		backOuter->add(IGeCurveBase::createSegment(
			GeSegment3d::create(
				backStart + m_vtAxisY * m_dR / 2,
				backStart - m_vtAxisY * m_dR / 2)
		));

		GeCurveArrayPtr backInner = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Inner);
		backInner->add(IGeCurveBase::createEllipse(GeEllipse3d::createByCircle(backStart + m_vtAxisX * m_dAL, m_vtAxisZ, m_dR / 4)));
		backCurve->add(backOuter);
		backCurve->add(backInner);

		// 后端法兰
		GeExtrusionInfo backExtrusionInfo(backCurve, m_vtAxisZ * m_dR * 0.2, true);
		IGeSolidBasePtr backExtrusion = IGeSolidBase::createGeExtrusion(backExtrusionInfo);
		backExtrusion->transform(trans);
		graphicsPtr->addGeSolidBase(*backExtrusion, sym1, m_dAlpha);
	}

	return graphicsPtr;
}

