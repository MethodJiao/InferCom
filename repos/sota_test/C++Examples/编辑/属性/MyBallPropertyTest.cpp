#include "stdafx.h"
#include "MyBallPropertyTest.h"

using namespace TestObject;
MyBallPropertyTest::MyBallPropertyTest()
{

}

MyBallPropertyTest::~MyBallPropertyTest()
{

}

void MyBallPropertyTest::OnPropertyGet(std::vector<BPEntityP> const& refps, PBBimUIProperyList& lst)
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

	PBBimUIPropertyItem properties[BallPropName::BallPropCount];

	for (int i = 0; i < pbObjs.size(); ++i)
	{
		IBPObjectPtr ptrOobject = pbObjs.at(i);
		BallTestP pBall = dynamic_cast<BallTestP>(objectPtr.get());
		if (NULL == pBall)
			continue;

		GePoint3d ptOri = pBall->getOrigin();
		// ----------------------------------------X---------------------------------------
		{	
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"圆心X", ptOri.x);
				properties[BallPropName::OriginX].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[BallPropName::OriginX];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double oldX;
					oldItem->getValue(oldX);
					if (fabs(oldX)- fabs(ptOri.x) > 0.0001)
						properties[BallPropName::OriginX].setMultiValue(true);
				}
			}
		}

		// ----------------------------------------Y---------------------------------------
		{
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"圆心Y", ptOri.y);
				properties[BallPropName::OriginY].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[BallPropName::OriginY];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double old;
					oldItem->getValue(old);
					if (fabs(old) - fabs(ptOri.y) > 0.0001)
						properties[BallPropName::OriginY].setMultiValue(true);
				}
			}
		}
		// ----------------------------------------Z---------------------------------------
		{
			if (0 == i)
			{
				PBBimUIPropertyItem item(L"圆心Z", ptOri.z);
				properties[BallPropName::OriginZ].swap(item);
			}
			else
			{
				PBBimUIPropertyItem* oldItem = &properties[BallPropName::OriginZ];
				if (!oldItem->multiValue())     // 比较判断是否为多值
				{
					PBBimUIPropertyItem::ListValue oldType;
					double old;
					oldItem->getValue(old);
					if (fabs(old) - fabs(ptOri.z) > 0.0001)
						properties[BallPropName::OriginZ].setMultiValue(true);
				}
			}
		}
	}

	lst.AppendGroup(L"构件属性");
	int nIndex = 0;
	for (int i = 0; i < BallPropName::BallPropCount; ++i)
	{
		lst.Append(nIndex++, properties[i]);
	}

	T_Super::GetUserExtendProperty(pProject, pbObjs, nIndex, lst);
}

TIErrorStatus MyBallPropertyTest::OnPropertySet(std::vector<BPEntityP> const& refps, int index, PBBimUIPropertyItem const& item)
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
		IBPObjectPtr pbObjPtr = BPObjectExtensionManager::getInstance().getBPObject(*elementRef);
		if (!pbObjPtr.isValid())
			continue;
		pbObjs.push_back(pbObjPtr);
	}

	if (pbObjs.size() < 1)
		return TIErrorStatus::failed;
	if (index < BallPropCount)
	{
		for (int indexGj = 0; indexGj < pbObjs.size(); ++indexGj)
		{
			IBPObjectPtr objectPtr = pbObjs.at(indexGj);
			BallTestP pBall = dynamic_cast<BallTest*>(objectPtr.get());
			if (pBall == nullptr)
				continue;

			GePoint3d ptOri = pBall->getOrigin();
			switch (index)
			{
			case BallPropName::OriginX:
			{
				double dValue = 0;
				item.getValue(dValue);

				ptOri.x = dValue;
				pBall->setOrigin(ptOri);
			}
			break;
			case BallPropName::OriginY:
			{
				double dValue = 0;
				item.getValue(dValue);

				ptOri.y = dValue;
				pBall->setOrigin(ptOri);
			}
			break;
			case BallPropName::OriginZ:
			{
				double dValue = 0;
				item.getValue(dValue);

				ptOri.z = dValue;
				pBall->setOrigin(ptOri);
			}
			break;
			default:
				break;

			}
			pBall->replaceInProject(*pProject);

		}
		return TIErrorStatus::succeed;
	}
	else
	{
		return T_Super::SetUserExtendProperty(pProject, pbObjs, index, item);
	}
}

class BallPropertyFactoryTest :public IToolInterfaceFactory
{
public:
	virtual IToolInterface* CreateInterface() override
	{
		MyBallPropertyTest* p = new MyBallPropertyTest();
		p->AddRef();
		return p;
	}
};

static BallPropertyFactoryTest s_BallPropertyFactory;

AutoDoRegisterFunctionsBegin
PBBimToolsInterfaceManager::RegisterFactory("BallTest", IToolNameProperty, &s_BallPropertyFactory);
AutoDoRegisterFunctionsEnd