
namespace VertoTest.Models
{
    public class EditEventRequest
    {
        public string Token { get; set; } // Case-sensitive, ensure it matches "token"
        public EventData EventData { get; set; } // This should match the structure of the JSON data for the updated event
    }
}
