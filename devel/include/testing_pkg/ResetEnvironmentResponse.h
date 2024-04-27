// Generated by gencpp from file testing_pkg/ResetEnvironmentResponse.msg
// DO NOT EDIT!


#ifndef TESTING_PKG_MESSAGE_RESETENVIRONMENTRESPONSE_H
#define TESTING_PKG_MESSAGE_RESETENVIRONMENTRESPONSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace testing_pkg
{
template <class ContainerAllocator>
struct ResetEnvironmentResponse_
{
  typedef ResetEnvironmentResponse_<ContainerAllocator> Type;

  ResetEnvironmentResponse_()
    : success(false)  {
    }
  ResetEnvironmentResponse_(const ContainerAllocator& _alloc)
    : success(false)  {
  (void)_alloc;
    }



   typedef uint8_t _success_type;
  _success_type success;





  typedef boost::shared_ptr< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> const> ConstPtr;

}; // struct ResetEnvironmentResponse_

typedef ::testing_pkg::ResetEnvironmentResponse_<std::allocator<void> > ResetEnvironmentResponse;

typedef boost::shared_ptr< ::testing_pkg::ResetEnvironmentResponse > ResetEnvironmentResponsePtr;
typedef boost::shared_ptr< ::testing_pkg::ResetEnvironmentResponse const> ResetEnvironmentResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator1> & lhs, const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator2> & rhs)
{
  return lhs.success == rhs.success;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator1> & lhs, const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace testing_pkg

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "358e233cde0c8a8bcfea4ce193f8fc15";
  }

  static const char* value(const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x358e233cde0c8a8bULL;
  static const uint64_t static_value2 = 0xcfea4ce193f8fc15ULL;
};

template<class ContainerAllocator>
struct DataType< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "testing_pkg/ResetEnvironmentResponse";
  }

  static const char* value(const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "bool success\n"
;
  }

  static const char* value(const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.success);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ResetEnvironmentResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::testing_pkg::ResetEnvironmentResponse_<ContainerAllocator>& v)
  {
    s << indent << "success: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.success);
  }
};

} // namespace message_operations
} // namespace ros

#endif // TESTING_PKG_MESSAGE_RESETENVIRONMENTRESPONSE_H
