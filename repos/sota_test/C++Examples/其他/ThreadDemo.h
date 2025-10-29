#pragma once
/** @class
*  @brief   多线程范例
*  @author  北京构力
*  ------------------------------------------------------------
*  版本历史       注释                       日期
*  ------------------------------------------------------------
*  @version v1.0  初始版本              2020/4/26
*  ------------------------------------------------------------
*  @note:  -
*/


class ThreadFunListener
{
public:
	static ThreadFunListener& Get();
	ThreadFunListener() {}
	virtual ~ThreadFunListener() {}
	void BeginRecord();
	void EndRecord();
private:
	static ThreadFunListener* s_recordThread;
	BIMBase::IBPThreadJob* m_threadP;

};


