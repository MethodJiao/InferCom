#include "stdafx.h"
#include "ToolSelectDemo.h"

using namespace ::p3d::platform;

ToolSelectDemo::ToolSelectDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	
}


ToolSelectDemo::~ToolSelectDemo()
{
	BPSelectionSetManager::getInstance().emptyAll();  // 清空选择集
	getEntityArray()->clear();
}


void ToolSelectDemo::_onPostInstall()
{
	T_Super::_onPostInstall();
	_setLocateCursor(true);
	BPSnap::getInstance().enableSnap(true);	
	BPUserInputManager::outPutPromptToCommandBar(Utf8String(L"请选择构件"));
}


void ToolSelectDemo::_setupAndPromptForNextAction()
{

}

void   ToolSelectDemo::_onRestartTool()
{
	ToolSelectDemo* newTool = new ToolSelectDemo();
	newTool->installTool();
}

bool ToolSelectDemo::_onDataButton( BPBaseButtonEventCP ev)
{
	__super::_onDataButton(ev);
	BIMBase::Core::IBPProjectManagerP pProjectManager = BIMBase::Core::BPApplication::getInstance().getProjectManager();
	if (pProjectManager == nullptr)
		return false;
	BPProjectPtr ptrProject = pProjectManager->getMainProject();
	if (ptrProject.isNull())
		return false;
	CString sName;
	bool bFlg = false;
	for (size_t i = 0; i < getEntityArray()->getCount(); i++)
	{
		BPEntityPtr ptrCurSel = getEntityArray()->getByIndex(i);
		if (ptrCurSel == nullptr)
			continue;		

		bFlg = true;
		//---------------获取鼠标当前选中的面------------------------------------------
		//获取solid的面
		IGeometryPtr ptrGeometry =  BPEntityLocateManager::getInstance().pickSolidGeomFace(*ev->getPoint(), ev->getViewport(), *ptrCurSel);
		if (ptrGeometry != NULL)
		{
			GeCurveArrayPtr ptrCurveSelect = ptrGeometry->getAsGeCurveArray();
		}
		else
		{
			//获取三角面片的面
			pvector<int> geoIndex;//获取到的三角面片在polyface中起始点index
			P3DStatus status = BPEntityLocateManager::getInstance().doLocateEntityGeom(BPPickGeomType::enSurface, *ev->getPoint(), ev->getViewport(), *ptrCurSel, geoIndex, false, 0);
			if (status == P3DStatus::ERROR)
				continue;
		
			BPGraphicsPtr ptrGraphicSel = BPGraphics::getGraphicsFromEntity(*ptrCurSel);
			if (ptrGraphicSel.isNull())
				continue;
			if (geoIndex.size() == 0)
				continue;
			pvector<GePoint3d> facePoints;//从polyface获取选中面片的点		
			for (BPGraphics::EntryPtr& loadedEntry : *ptrGraphicSel)
			{
				switch (loadedEntry->getType())
				{
				case BPGraphics::Entry::Type::Polyface:
				{
					PolyfaceHandleP pPolyface = loadedEntry->getAsPolyfaceHandleP();
					if (pPolyface == nullptr)
						continue;

					TemplateVector<GePoint3d> points = pPolyface->getPointR();
					TemplateVectorIntR index = pPolyface->getPointIndexR();
					for (int j = geoIndex[0]; j < index.size(); j++)
					{
						if (index[j] != 0)//0作为三角面片中各面的点的分隔符
							facePoints.push_back(points[index[j]]);
						else
							break;
					}
				}
				}
			}
		}
		//------------------------------------------------------------------------------------------------
		
		BPDataPtr ptrInstance = BPDataUtil::getDataOnEntity(*ptrCurSel);
		if (!ptrInstance.isValid())
			continue;

		IBPObjectPtr ptrCopy = BPObjectExtensionManager::getInstance().getBPObject(*ptrCurSel);
		if (!ptrCopy.isValid())
			continue;

		Utf8String sClassName = ptrCopy->getClassName();
		sName += sClassName.c_str();
		sName += L";";
	}

	if(bFlg)
		AfxMessageBox(L"已获取信息");
	else
		AfxMessageBox(L"请选择对象");

	Utf8String ustTemp;
	P3DStringHelper::wCharToUtf8(ustTemp, sName);
	Json::Value jsonMessage = ustTemp;
	P3DGlobalVariableManager::getManager().setValue("Demo_MESSAGE", jsonMessage);
	return true;
}


void ToolSelectDemo::_onDynamicFrame( BPBaseButtonEventCP ev)
{
	if (NULL == ev)
		return;
}
static int s_index = 1;
bool ToolSelectDemo::_onResetButton( BPBaseButtonEventCP ev)
{
	_exitTool();
	return true;
}

bool ToolSelectDemo::_onKeyTransition(bool bWentDown, ::p3d::platform::P3DVirtualKey key, bool shiftIsDown, bool ctrlIsDown)
{
	return true;
}

bool ToolSelectDemo::_onModifierKeyTransition(bool bWentDown, int key)
{
	return __super::_onModifierKeyTransition(bWentDown, key);
}


BPTool* CreateDemoSelectTool()
{
	ToolSelectDemo* tool = new ToolSelectDemo();
	return tool;
	return NULL;
}



AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("selectToolDemo", &CreateDemoSelectTool);
AutoDoRegisterFunctionsEnd